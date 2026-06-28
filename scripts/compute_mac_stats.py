#!/usr/bin/env python3
"""Compute per-MAC-type mean/std from collected MAC output distributions.

Reads mac_dist_results/<run>/mac_distribution.csv (long format:
mac_type, layer_idx, value, count) and writes a JSON mapping each MAC type to
its pooled-over-layers mean and std (in the integer MAC-output domain), plus a
per-layer breakdown. This JSON feeds eval_mac_transform_glue.py.

Stats for a given MAC type are identical across runs (same noise-free network,
same data), so when a type appears in several run folders we keep the first
occurrence (preferring the single-type folder named exactly after the type).
"""

from __future__ import division, print_function

import argparse
import csv
import json
import math
import os
from collections import defaultdict


def read_distribution_csv(path):
    """Return mac_type -> layer_idx -> {value: count}."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            data[r['mac_type']][int(r['layer_idx'])][int(r['value'])] = int(r['count'])
    return data


def mean_std(value_count):
    total = sum(value_count.values())
    if total == 0:
        return 0.0, 0.0, 0
    mean = sum(v * c for v, c in value_count.items()) / total
    var = sum(c * (v - mean) ** 2 for v, c in value_count.items()) / total
    return mean, math.sqrt(var), total


def main():
    here = os.path.dirname(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description='Compute per-MAC-type mean/std')
    parser.add_argument('--results_dir', default=os.path.join(here, 'mac_dist_results'),
                        help='Directory with per-run subfolders containing mac_distribution.csv')
    parser.add_argument('--output', default=os.path.join(here, 'mac_dist_results', 'mac_stats.json'),
                        help='Where to write the stats JSON')
    args = parser.parse_args()

    entries = sorted(os.listdir(args.results_dir))
    # Prefer single-type folders (folder name == one MAC type) first.
    folders = sorted(entries, key=lambda e: (',' in e, e))

    stats = {}
    for entry in folders:
        csv_path = os.path.join(args.results_dir, entry, 'mac_distribution.csv')
        if not os.path.isfile(csv_path):
            continue
        data = read_distribution_csv(csv_path)
        for mac_type, layers in data.items():
            if mac_type in stats:
                continue
            pooled = layers.get(-1, {})
            mean, std, n = mean_std(pooled)
            per_layer = {}
            for layer_idx, vc in layers.items():
                if layer_idx < 0:
                    continue
                lm, ls, ln = mean_std(vc)
                per_layer[str(layer_idx)] = {'mean': lm, 'std': ls, 'n': ln}
            stats[mac_type] = {
                'mean': mean, 'std': std, 'n': n,
                'source': entry, 'per_layer': per_layer,
            }

    if not stats:
        raise SystemExit('No mac_distribution.csv files found under %s' % args.results_dir)

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print('Wrote stats for %d MAC types to %s' % (len(stats), args.output))
    print('%-14s %12s %10s %14s' % ('mac_type', 'mean', 'std', 'N'))
    for mac_type in sorted(stats):
        s = stats[mac_type]
        print('%-14s %12.3f %10.3f %14d' % (mac_type, s['mean'], s['std'], s['n']))


if __name__ == '__main__':
    main()
