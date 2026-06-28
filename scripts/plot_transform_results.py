#!/usr/bin/env python3
"""Plot MAC-output transform sweeps from transform_results/.

Layout expected:
  transform_results/<mode>/<tag>/transform_sweep_results.csv
where <mode> is 'shift' or 'clamp' and <tag> is the MAC-type combination.

For each mode it draws a comparison of accuracy (and eval loss) vs the
multiplier x across all configs, plus an individual plot per config, and a
summary CSV.
"""

from __future__ import division, print_function

import argparse
import csv
import os
import re

import matplotlib.pyplot as plt

ALL_TYPES = 'qkv,attn_score,attn_apply,output_proj,ffn1,ffn2'


def slugify(name):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', name).strip('_')


def pretty_tag(tag):
    return 'all' if tag == ALL_TYPES else tag


def read_sweep_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    x = [float(r['x']) for r in rows]
    acc = [float(r['acc']) * 100.0 for r in rows]
    acc_mm = [float(r['acc_mm']) * 100.0 for r in rows if r.get('acc_mm', '') != '']
    eval_loss = [float(r['eval_loss']) for r in rows]
    return {'x': x, 'acc': acc, 'acc_mm': acc_mm, 'eval_loss': eval_loss}


def discover_runs(results_dir):
    runs = []
    for mode in sorted(os.listdir(results_dir)):
        mode_dir = os.path.join(results_dir, mode)
        if not os.path.isdir(mode_dir):
            continue
        for tag in sorted(os.listdir(mode_dir)):
            csv_path = os.path.join(mode_dir, tag, 'transform_sweep_results.csv')
            if os.path.isfile(csv_path):
                data = read_sweep_csv(csv_path)
                data.update(mode=mode, tag=tag, label=pretty_tag(tag))
                runs.append(data)
    if not runs:
        raise SystemExit('No transform_sweep_results.csv files found under %s' % results_dir)
    return runs


def mode_title(mode):
    if mode == 'shift':
        return 'shift (int + x*std)'
    if mode == 'clamp':
        return 'clamp to mean +/- x*std'
    return mode


def plot_single_run(run, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('MNLI %s: %s' % (mode_title(run['mode']), run['label']), fontsize=12, y=1.02)

    axes[0].plot(run['x'], run['acc'], 'o-', color='#2563eb', linewidth=2, markersize=5, label='matched')
    if run['acc_mm']:
        axes[0].plot(run['x'], run['acc_mm'], 's--', color='#16a34a', linewidth=2, markersize=5, label='mismatched')
    axes[0].set_xlabel('x (multiplier of std)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Dev accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(run['x'], run['eval_loss'], 'o-', color='#dc2626', linewidth=2, markersize=5)
    axes[1].set_xlabel('x (multiplier of std)')
    axes[1].set_ylabel('Eval loss')
    axes[1].set_title('Cross-entropy loss')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    slug = '%s_%s' % (run['mode'], slugify(run['label']))
    png_path = os.path.join(out_dir, '%s.png' % slug)
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, '%s.pdf' % slug), bbox_inches='tight')
    plt.close(fig)
    return png_path


def plot_mode_comparison(mode, runs, out_dir):
    runs = [r for r in runs if r['mode'] == mode]
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('MNLI %s: accuracy vs x across MAC configurations' % mode_title(mode),
                 fontsize=13, y=1.02)

    colors = plt.cm.tab10(range(len(runs)))
    for run, color in zip(runs, colors):
        axes[0].plot(run['x'], run['acc'], 'o-', label=run['label'],
                     color=color, linewidth=2, markersize=4)
        axes[1].plot(run['x'], run['eval_loss'], 'o-', label=run['label'],
                     color=color, linewidth=2, markersize=4)

    axes[0].axhline(33.33, color='gray', linewidth=0.8, linestyle=':', label='chance (3-class)')
    axes[0].set_xlabel('x (multiplier of std)')
    axes[0].set_ylabel('Matched accuracy (%)')
    axes[0].set_title('Dev accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc='best')

    axes[1].set_xlabel('x (multiplier of std)')
    axes[1].set_ylabel('Eval loss')
    axes[1].set_title('Eval loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc='best')

    fig.tight_layout()
    png_path = os.path.join(out_dir, 'comparison_%s.png' % mode)
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'comparison_%s.pdf' % mode), bbox_inches='tight')
    plt.close(fig)
    return png_path


def write_summary(runs, out_dir):
    path = os.path.join(out_dir, 'summary.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'config', 'acc_x0', 'acc_xmax', 'x_max', 'best_acc', 'worst_acc'])
        for run in runs:
            writer.writerow([
                run['mode'], run['label'],
                '%.4f' % run['acc'][0], '%.4f' % run['acc'][-1], '%g' % run['x'][-1],
                '%.4f' % max(run['acc']), '%.4f' % min(run['acc']),
            ])
    return path


def main():
    here = os.path.dirname(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description='Plot MAC transform sweep results')
    parser.add_argument('--results_dir', default=os.path.join(here, 'transform_results'))
    parser.add_argument('--output_dir', default=os.path.join(here, 'transform_results_plots'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = discover_runs(args.results_dir)
    print('Found %d run(s):' % len(runs))
    for run in runs:
        print('  - %s / %s' % (run['mode'], run['label']))

    for run in runs:
        png = plot_single_run(run, args.output_dir)
        print('Wrote %s' % png)

    for mode in sorted({r['mode'] for r in runs}):
        png = plot_mode_comparison(mode, runs, args.output_dir)
        if png:
            print('Wrote %s' % png)

    summary = write_summary(runs, args.output_dir)
    print('Wrote %s' % summary)
    print('Done. Plots saved to %s' % args.output_dir)


if __name__ == '__main__':
    main()
