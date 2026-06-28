# coding=utf-8
"""In-DRAM binary MAC noise simulation for evaluation."""

import re

import torch

VALID_MAC_TYPES = frozenset({
    'qkv',
    'attn_score',
    'attn_apply',
    'output_proj',
    'ffn1',
    'ffn2',
})


class MacNoiseConfig(object):
    """Global eval-time config for noisy binary MAC injection."""

    enabled = False
    noise_pct = 0.0
    mac_types = frozenset()

    @classmethod
    def configure(cls, noise_pct, mac_types, enabled=True):
        unknown = set(mac_types) - VALID_MAC_TYPES
        if unknown:
            raise ValueError(
                'Unknown mac_types: %s. Valid: %s' % (unknown, sorted(VALID_MAC_TYPES)))
        cls.enabled = enabled
        cls.noise_pct = float(noise_pct)
        cls.mac_types = frozenset(mac_types)

    @classmethod
    def reset(cls):
        cls.enabled = False
        cls.noise_pct = 0.0
        cls.mac_types = frozenset()

    @classmethod
    def should_noise(cls, mac_type):
        return cls.enabled and mac_type is not None and mac_type in cls.mac_types

    @classmethod
    def sigma(cls, mac_size):
        """3-sigma rule: ~99.7%% of noise within +/- noise_pct%% of mac_size."""
        return (cls.noise_pct / 100.0) * float(mac_size) / 3.0

    @classmethod
    def inject(cls, integer_dot, mac_size, mac_type):
        if not cls.should_noise(mac_type):
            return integer_dot
        sigma = cls.sigma(mac_size)
        if sigma == 0.0:
            return integer_dot
        noise = torch.randn_like(integer_dot, dtype=torch.float32) * sigma
        return integer_dot.float() + noise


class MacOutputCollector(object):
    """Collect integer binary-MAC outputs for distribution analysis.

    A binary MAC output is the integer dot product of activation codes with
    weight signs, BEFORE any noise or scale is applied. We accumulate these
    integers into per-(mac_type, layer) histograms with integer bins. The
    value range is bounded by +/- mac_size, so bins stay small.
    """

    enabled = False
    mac_types = frozenset()
    _hists = {}

    @classmethod
    def configure(cls, mac_types, enabled=True):
        unknown = set(mac_types) - VALID_MAC_TYPES
        if unknown:
            raise ValueError(
                'Unknown mac_types: %s. Valid: %s' % (unknown, sorted(VALID_MAC_TYPES)))
        cls.enabled = enabled
        cls.mac_types = frozenset(mac_types)
        cls._hists = {}

    @classmethod
    def reset(cls):
        cls.enabled = False
        cls.mac_types = frozenset()
        cls._hists = {}

    @classmethod
    def should_collect(cls, mac_type):
        return cls.enabled and mac_type is not None and mac_type in cls.mac_types

    @classmethod
    def record(cls, mac_type, layer_idx, integer_output, mac_size):
        if not cls.should_collect(mac_type):
            return
        mac_size = int(mac_size)
        vals = integer_output.detach().reshape(-1).round().to(torch.long).cpu()
        shifted = (vals + mac_size).clamp_(0, 2 * mac_size)
        counts = torch.bincount(shifted, minlength=2 * mac_size + 1)
        key = (mac_type, int(layer_idx))
        entry = cls._hists.get(key)
        if entry is None:
            cls._hists[key] = {'mac_size': mac_size, 'counts': counts}
        else:
            entry['counts'] += counts

    @classmethod
    def export_rows(cls):
        """Long-format rows: (mac_type, layer_idx, value, count).

        layer_idx == -1 is the pooled-over-all-layers histogram. Only nonzero
        bins are emitted to keep the output compact.
        """
        rows = []
        pooled = {}
        for (mac_type, layer_idx), entry in sorted(cls._hists.items()):
            mac_size = entry['mac_size']
            counts = entry['counts']
            p = pooled.get(mac_type)
            if p is None:
                pooled[mac_type] = {'mac_size': mac_size, 'counts': counts.clone()}
            else:
                p['counts'] += counts
            nz = counts.nonzero(as_tuple=False).reshape(-1)
            for idx in nz.tolist():
                rows.append((mac_type, layer_idx, idx - mac_size, int(counts[idx].item())))
        for mac_type, p in sorted(pooled.items()):
            mac_size = p['mac_size']
            counts = p['counts']
            nz = counts.nonzero(as_tuple=False).reshape(-1)
            for idx in nz.tolist():
                rows.append((mac_type, -1, idx - mac_size, int(counts[idx].item())))
        return rows


_LAYER_RE = re.compile(r'\.layer\.(\d+)\.')


def register_mac_sites(model):
    """Tag MAC-bearing modules with their transformer layer index.

    Layer index is parsed from the module's registered name (e.g.
    'bert.encoder.layer.7.output.dense' -> 7). Modules outside the encoder
    stack (e.g. pooler) get -1. This avoids threading layer_idx through every
    constructor.
    """
    for name, module in model.named_modules():
        match = _LAYER_RE.search(name)
        layer_idx = int(match.group(1)) if match else -1
        if getattr(module, 'mac_type', None) is not None:
            module._mac_layer_idx = layer_idx
        if module.__class__.__name__ == 'BertSelfAttention':
            module._mac_layer_idx = layer_idx


def tensor_scalar(param):
    if param is None:
        return 1.0
    return float(param.reshape(-1)[0].detach())


def bwn_weight_scale(weight_fp, layerwise=True):
    if layerwise:
        return weight_fp.norm(p=1).div(weight_fp.nelement())
    n = weight_fp[0].nelement()
    return weight_fp.norm(1, 1, keepdim=True).div(n)


def bwn_weight_sign(weight_fp, layerwise=True):
    e = weight_fp.mean()
    sign_w = (weight_fp - e).sign()
    if not layerwise:
        raise NotImplementedError('Per-row BWN sign not implemented for noise sim')
    return sign_w


def activation_binary_codes(input_moved, input_q, clip_val, symmetric, input_bits):
    """Binary activation codes for integer MAC simulation.

    - symmetric (Q/K/V, ffn1, output_proj, ...): codes in {-1, 0, +1}
    - asymmetric unsigned (ffn2, attention probs): codes in {0, 1}
    """
    if input_bits >= 32:
        return input_moved.sign()
    if symmetric:
        return input_moved.sign()
    alpha = clip_val.reshape(()).to(device=input_q.device, dtype=input_q.dtype)
    eps = torch.tensor(1e-5, device=input_q.device, dtype=input_q.dtype)
    alpha = torch.where(alpha > eps, alpha, eps)
    return (input_q / alpha).round().clamp(0, 1)


def noisy_binary_linear(sign_input, sign_weight, mac_size, mac_type, input_scale, weight_scale):
    """Integer dot product on signs, optional noise, then apply scales."""
    int_dot = torch.nn.functional.linear(sign_input, sign_weight)
    int_dot = MacNoiseConfig.inject(int_dot, mac_size, mac_type)
    return input_scale * weight_scale * int_dot


def noisy_binary_matmul(sign_a, sign_b, mac_size, mac_type, scale_a, scale_b, transpose_b=False):
    if transpose_b:
        int_out = torch.matmul(sign_a, sign_b.transpose(-1, -2))
    else:
        int_out = torch.matmul(sign_a, sign_b)
    int_out = MacNoiseConfig.inject(int_out, mac_size, mac_type)
    return scale_a * scale_b * int_out


class MacTransform(object):
    """Deterministic, noise-free transforms applied to integer MAC outputs.

    Operates in the same integer-MAC domain as MacOutputCollector, using
    per-mac-type statistics (mean, std) measured from the noise-free network.
    No Gaussian noise is ever applied here.

    Modes:
      - 'shift': out = int_out + multiplier * std[mac_type]
      - 'clamp': out = clip(int_out, mean - multiplier*std, mean + multiplier*std)
    """

    enabled = False
    mode = 'off'
    multiplier = 0.0
    mac_types = frozenset()
    stats = {}

    @classmethod
    def configure(cls, mode, multiplier, mac_types, stats, enabled=True):
        if mode not in ('shift', 'clamp'):
            raise ValueError("mode must be 'shift' or 'clamp', got %r" % mode)
        unknown = set(mac_types) - VALID_MAC_TYPES
        if unknown:
            raise ValueError(
                'Unknown mac_types: %s. Valid: %s' % (unknown, sorted(VALID_MAC_TYPES)))
        missing = [t for t in mac_types if t not in stats]
        if missing:
            raise ValueError('Missing stats (mean/std) for mac_types: %s' % missing)
        cls.enabled = enabled
        cls.mode = mode
        cls.multiplier = float(multiplier)
        cls.mac_types = frozenset(mac_types)
        cls.stats = dict(stats)

    @classmethod
    def reset(cls):
        cls.enabled = False
        cls.mode = 'off'
        cls.multiplier = 0.0
        cls.mac_types = frozenset()
        cls.stats = {}

    @classmethod
    def should_apply(cls, mac_type):
        return cls.enabled and mac_type is not None and mac_type in cls.mac_types

    @classmethod
    def apply(cls, integer_output, mac_type):
        if not cls.should_apply(mac_type):
            return integer_output
        st = cls.stats[mac_type]
        mean = float(st['mean'])
        std = float(st['std'])
        if cls.mode == 'shift':
            return integer_output.float() + cls.multiplier * std
        if cls.mode == 'clamp':
            lo = mean - cls.multiplier * std
            hi = mean + cls.multiplier * std
            return integer_output.float().clamp(min=lo, max=hi)
        return integer_output


def transformed_binary_matmul(sign_a, sign_b, mac_type, scale_a, scale_b, transpose_b=False):
    """Integer matmul on signs, apply MacTransform, then rescale (no noise)."""
    if transpose_b:
        int_out = torch.matmul(sign_a, sign_b.transpose(-1, -2))
    else:
        int_out = torch.matmul(sign_a, sign_b)
    int_out = MacTransform.apply(int_out, mac_type)
    return scale_a * scale_b * int_out
