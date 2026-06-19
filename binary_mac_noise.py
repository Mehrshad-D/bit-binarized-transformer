# coding=utf-8
"""In-DRAM binary MAC noise simulation for evaluation."""

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
