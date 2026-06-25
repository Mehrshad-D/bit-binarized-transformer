#!/usr/bin/env python3
"""Plot binary-MAC output distributions from mac_dist_results/.

Reads the long-format CSV written by collect_mac_distribution.py
(columns: mac_type, layer_idx, value, count) and, for each run directory,
draws per-MAC-type histograms (pooled over all layers plus a per-layer
overlay) and a normalized cross-type comparison.
"""

from __future__ import division, print_function

import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


def slugify(name):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', name).strip('_')


def read_distribution_csv(path):
    """Return nested dict: mac_type -> layer_idx -> {value: count}."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            mac_type = r['mac_type']
            layer_idx = int(r['layer_idx'])
            value = int(r['value'])
            count = int(r['count'])
            data[mac_type][layer_idx][value] = count
    return data


def sorted_arrays(value_count):
    items = sorted(value_count.items())
    values = [v for v, _ in items]
    counts = [c for _, c in items]
    return values, counts


def weighted_mean_std(value_count):
    total = sum(value_count.values())
    if total == 0:
        return 0.0, 0.0
    mean = sum(v * c for v, c in value_count.items()) / total
    var = sum(c * (v - mean) ** 2 for v, c in value_count.items()) / total
    return mean, math.sqrt(var)


def normalized(counts):
    total = float(sum(counts))
    if total == 0:
        return counts
    return [c / total for c in counts]


def plot_mac_type(mac_type, layers, out_dir):
    """layers: dict layer_idx -> {value: count}. -1 is the pooled histogram."""
    pooled = layers.get(-1, {})
    per_layer = {k: v for k, v in layers.items() if k >= 0}

    mean, std = weighted_mean_std(pooled)
    n_samples = sum(pooled.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        'Binary MAC output distribution: %s  (mean=%.1f, std=%.1f, N=%d)'
        % (mac_type, mean, std, n_samples), fontsize=13, y=1.02)

    # Left: pooled distribution over all layers.
    values, counts = sorted_arrays(pooled)
    axes[0].fill_between(values, normalized(counts), step='mid', alpha=0.4, color='#2563eb')
    axes[0].plot(values, normalized(counts), drawstyle='steps-mid', color='#2563eb', linewidth=1.5)
    axes[0].axvline(mean, color='#dc2626', linewidth=1, linestyle='--', label='mean')
    axes[0].set_xlabel('Integer MAC output')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pooled over all layers')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Right: one curve per layer.
    colors = plt.cm.viridis([i / max(len(per_layer) - 1, 1) for i in range(len(per_layer))])
    for (layer_idx, vc), color in zip(sorted(per_layer.items()), colors):
        lv, lc = sorted_arrays(vc)
        axes[1].plot(lv, normalized(lc), color=color, linewidth=1.0, alpha=0.8,
                     label='layer %d' % layer_idx)
    axes[1].set_xlabel('Integer MAC output')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Per-layer')
    axes[1].grid(True, alpha=0.3)
    if per_layer:
        axes[1].legend(fontsize=6, ncol=2, loc='best')

    fig.tight_layout()
    slug = slugify(mac_type)
    png_path = os.path.join(out_dir, '%s.png' % slug)
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, '%s.pdf' % slug), bbox_inches='tight')
    plt.close(fig)
    return png_path


def plot_comparison(data, out_dir):
    """Overlay pooled distributions of all MAC types, x normalized to [-1, 1].

    MAC types have different output ranges (e.g. attn_score spans +/-64,
    ffn2 spans +/-3072), so the x-axis is normalized by each type's observed
    max-abs output to make the shapes comparable.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(range(len(data)))
    for (mac_type, layers), color in zip(sorted(data.items()), colors):
        pooled = layers.get(-1, {})
        if not pooled:
            continue
        scale = max(abs(v) for v in pooled) or 1
        values, counts = sorted_arrays(pooled)
        xs = [v / scale for v in values]
        ax.plot(xs, normalized(counts), color=color, linewidth=1.5, alpha=0.8,
                label='%s (range +/-%d)' % (mac_type, scale))
    ax.set_xlabel('Integer MAC output / max-abs output')
    ax.set_ylabel('Frequency')
    ax.set_title('MAC output distributions (normalized x), pooled over layers')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    png_path = os.path.join(out_dir, 'comparison_mac_types.png')
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'comparison_mac_types.pdf'), bbox_inches='tight')
    plt.close(fig)
    return png_path


def discover_runs(results_dir):
    runs = []
    for entry in sorted(os.listdir(results_dir)):
        subdir = os.path.join(results_dir, entry)
        csv_path = os.path.join(subdir, 'mac_distribution.csv')
        if os.path.isdir(subdir) and os.path.isfile(csv_path):
            runs.append((entry, csv_path))
    if not runs:
        raise SystemExit('No mac_distribution.csv files found under %s' % results_dir)
    return runs


def main():
    here = os.path.dirname(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description='Plot MAC output distributions')
    parser.add_argument('--results_dir', default=os.path.join(here, 'mac_dist_results'),
                        help='Directory with per-run subfolders containing mac_distribution.csv')
    parser.add_argument('--output_dir', default=os.path.join(here, 'mac_dist_plots'),
                        help='Directory to write plot files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = discover_runs(args.results_dir)
    print('Found %d run(s):' % len(runs))

    for tag, csv_path in runs:
        data = read_distribution_csv(csv_path)
        run_out = os.path.join(args.output_dir, slugify(tag))
        os.makedirs(run_out, exist_ok=True)
        print('  - %s: %s' % (tag, ', '.join(sorted(data.keys()))))
        for mac_type, layers in data.items():
            png = plot_mac_type(mac_type, layers, run_out)
            print('    wrote %s' % png)
        if len(data) > 1:
            cmp_png = plot_comparison(data, run_out)
            print('    wrote %s' % cmp_png)

    print('Done. Plots saved to %s' % args.output_dir)


if __name__ == '__main__':
    main()
