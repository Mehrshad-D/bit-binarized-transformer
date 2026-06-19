#!/usr/bin/env python3
"""Plot MNLI noise sweep CSVs from noise_results/."""

from __future__ import division, print_function

import argparse
import csv
import os
import re

import matplotlib.pyplot as plt


def slugify(name):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', name).strip('_')


def read_sweep_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    noise_pct = [float(r['noise_pct']) for r in rows]
    acc = [float(r['acc']) * 100.0 for r in rows]
    acc_mm = [float(r['acc_mm']) * 100.0 for r in rows]
    eval_loss = [float(r['eval_loss']) for r in rows]
    mac_types = rows[0].get('noise_mac_types', os.path.basename(os.path.dirname(path)))
    return {
        'noise_pct': noise_pct,
        'acc': acc,
        'acc_mm': acc_mm,
        'eval_loss': eval_loss,
        'mac_types': mac_types,
        'path': path,
    }


def discover_runs(results_dir):
    runs = []
    for entry in sorted(os.listdir(results_dir)):
        subdir = os.path.join(results_dir, entry)
        csv_path = os.path.join(subdir, 'noise_sweep_results.csv')
        if os.path.isdir(subdir) and os.path.isfile(csv_path):
            data = read_sweep_csv(csv_path)
            data['label'] = entry
            data['slug'] = slugify(entry)
            runs.append(data)
    if not runs:
        raise SystemExit('No noise_sweep_results.csv files found under %s' % results_dir)
    return runs


def plot_single_run(data, out_dir):
    label = data['label']
    slug = data['slug']
    x = data['noise_pct']
    baseline_acc = data['acc'][0]
    delta = [v - baseline_acc for v in data['acc']]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('MNLI noise sweep: %s' % label, fontsize=13, y=1.02)

    axes[0].plot(x, data['acc'], 'o-', label='matched', color='#2563eb', linewidth=2, markersize=6)
    axes[0].plot(x, data['acc_mm'], 's--', label='mismatched', color='#16a34a', linewidth=2, markersize=6)
    axes[0].set_xlabel('Noise level (% of MAC size)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Dev accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, data['eval_loss'], 'o-', color='#dc2626', linewidth=2, markersize=6)
    axes[1].set_xlabel('Noise level (% of MAC size)')
    axes[1].set_ylabel('Eval loss')
    axes[1].set_title('Cross-entropy loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, delta, 'o-', color='#d97706', linewidth=2, markersize=6)
    axes[2].axhline(0.0, color='gray', linewidth=0.8, linestyle='--')
    axes[2].set_xlabel('Noise level (% of MAC size)')
    axes[2].set_ylabel('Δ accuracy (pp)')
    axes[2].set_title('Matched acc drop from 0% noise')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(out_dir, '%s.png' % slug)
    pdf_path = os.path.join(out_dir, '%s.pdf' % slug)
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def plot_comparison(runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MNLI: compare noise sensitivity across MAC configurations', fontsize=13, y=1.02)

    colors = plt.cm.tab10(range(len(runs)))
    for run, color in zip(runs, colors):
        axes[0].plot(
            run['noise_pct'], run['acc'], 'o-', label=run['label'],
            color=color, linewidth=2, markersize=5)
        axes[1].plot(
            run['noise_pct'], run['eval_loss'], 'o-', label=run['label'],
            color=color, linewidth=2, markersize=5)

    axes[0].set_xlabel('Noise level (% of MAC size)')
    axes[0].set_ylabel('Matched accuracy (%)')
    axes[0].set_title('Matched dev accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc='best')

    axes[1].set_xlabel('Noise level (% of MAC size)')
    axes[1].set_ylabel('Eval loss')
    axes[1].set_title('Eval loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc='best')

    fig.tight_layout()
    png_path = os.path.join(out_dir, 'comparison_all_runs.png')
    pdf_path = os.path.join(out_dir, 'comparison_all_runs.pdf')
    fig.savefig(png_path, dpi=160, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def write_summary_table(runs, out_dir):
  path = os.path.join(out_dir, 'summary.csv')
  with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'config', 'mac_types', 'acc_0pct', 'acc_20pct', 'drop_pp_0_to_20',
        'acc_mm_0pct', 'acc_mm_20pct', 'eval_loss_20pct'])
    for run in runs:
      acc0, acc20 = run['acc'][0], run['acc'][-1]
      mm0, mm20 = run['acc_mm'][0], run['acc_mm'][-1]
      writer.writerow([
          run['label'], run['mac_types'],
          '%.4f' % acc0, '%.4f' % acc20, '%.4f' % (acc20 - acc0),
          '%.4f' % mm0, '%.4f' % mm20,
          '%.4f' % run['eval_loss'][-1],
      ])
  return path


def main():
    parser = argparse.ArgumentParser(description='Plot noise sweep results')
    parser.add_argument(
        '--results_dir',
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'noise_results'),
        help='Directory containing per-config subfolders with noise_sweep_results.csv')
    parser.add_argument(
        '--output_dir',
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'noise_results_plots'),
        help='Directory to write plot files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = discover_runs(args.results_dir)

    print('Found %d configurations:' % len(runs))
    for run in runs:
        print('  -', run['label'])

    for run in runs:
        png, pdf = plot_single_run(run, args.output_dir)
        print('Wrote %s' % png)

    cmp_png, cmp_pdf = plot_comparison(runs, args.output_dir)
    print('Wrote %s' % cmp_png)

    summary_path = write_summary_table(runs, args.output_dir)
    print('Wrote %s' % summary_path)
    print('Done. Plots saved to %s' % args.output_dir)


if __name__ == '__main__':
    main()
