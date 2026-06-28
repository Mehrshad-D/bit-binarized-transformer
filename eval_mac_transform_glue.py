# coding=utf-8
"""Evaluate GLUE checkpoints under deterministic MAC-output transforms.

Two noise-free scenarios on the integer binary-MAC outputs, applied per MAC
type using stats measured from the noise-free network (mac_stats.json):

  - shift: out = int_out + x * std[mac_type]
  - clamp: out = clip(int_out, mean - x*std, mean + x*std)

Sweeps the multiplier x and writes accuracy to a CSV. Gaussian noise is never
enabled here (MacNoiseConfig stays off).
"""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import csv
import json
import logging
import os
import random

import torch
from torch.utils.data import DataLoader, SequentialSampler

from binary_mac_noise import MacNoiseConfig, MacTransform, VALID_MAC_TYPES
from helper import init_logging, print_args
from kd_learner_glue import KDLearner
from utils_glue import (
    convert_examples_to_features,
    default_params,
    get_tensor_data,
    output_modes,
    processors,
)
from transformer.configuration_bert import BertConfig
from transformer.modeling_bert_quant import BertForSequenceClassification
from transformer.tokenization import BertTokenizer


def parse_sweep(sweep_str):
    return [float(x.strip()) for x in sweep_str.split(',') if x.strip()]


def parse_mac_types(mac_types_str):
    mac_types = [x.strip() for x in mac_types_str.split(',') if x.strip()]
    if not mac_types:
        raise ValueError('--mac_types must list at least one MAC type')
    unknown = set(mac_types) - set(VALID_MAC_TYPES)
    if unknown:
        raise ValueError('Unknown MAC types: %s. Valid: %s'
                         % (sorted(unknown), sorted(VALID_MAC_TYPES)))
    return mac_types


def load_stats(stats_file, mac_types):
    with open(stats_file) as f:
        raw = json.load(f)
    stats = {}
    for t in mac_types:
        if t not in raw:
            raise ValueError('stats file %s has no entry for MAC type %r' % (stats_file, t))
        stats[t] = {'mean': float(raw[t]['mean']), 'std': float(raw[t]['std'])}
    return stats


def evaluate_once(learner, task_name, output_mode, eval_labels, num_labels,
                  eval_dataloader, mm_eval_dataloader, mm_eval_labels):
    learner.student_model.eval()
    result = learner._do_eval(
        learner.student_model, task_name, eval_dataloader, output_mode, eval_labels, num_labels)

    mm_acc = None
    if task_name == 'mnli' and mm_eval_dataloader is not None:
        mm_result = learner._do_eval(
            learner.student_model, 'mnli-mm', mm_eval_dataloader, output_mode, mm_eval_labels, num_labels)
        mm_acc = mm_result.get('acc', None)

    return result, mm_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--student_model', required=True, type=str,
                        help='Path to trained checkpoint dir (e.g. outputs/MNLI/W1A1/kd_joint)')
    parser.add_argument('--vocab_dir', required=True, type=str)
    parser.add_argument('--task_name', default='MNLI', type=str)
    parser.add_argument('--output_dir', default='./transform_results', type=str)
    parser.add_argument('--job_id', default='mac_transform', type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--max_seq_length', default=None, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--transform_mode', required=True, choices=['shift', 'clamp'],
                        help="'shift' adds x*std; 'clamp' clips to mean +/- x*std")
    parser.add_argument('--x_sweep', default='0,0.5,1,2,3', type=str,
                        help='Comma-separated multipliers x of std')
    parser.add_argument('--mac_types', required=True, type=str,
                        help='Comma-separated MAC types: %s' % ', '.join(sorted(VALID_MAC_TYPES)))
    parser.add_argument('--stats_file', default='./mac_dist_results/mac_stats.json', type=str,
                        help='JSON with per-MAC-type mean/std (from compute_mac_stats.py)')

    # quantization params (must match training checkpoint)
    parser.add_argument('--weight_bits', default=1, type=int)
    parser.add_argument('--input_bits', default=1, type=int)
    parser.add_argument('--weight_quant_method', default='bwn', type=str)
    parser.add_argument('--input_quant_method', default='elastic', type=str)
    parser.add_argument('--learnable_scaling', action='store_true', default=True)
    parser.add_argument('--ACT2FN', default='relu', type=str)
    parser.add_argument('--sym_quant_ffn_attn', action='store_true')
    parser.add_argument('--sym_quant_qkvo', action='store_true', default=True)
    parser.add_argument('--embed_layerwise', default=False, type=lambda x: bool(int(x)))
    parser.add_argument('--weight_layerwise', default=True, type=lambda x: bool(int(x)))
    parser.add_argument('--input_layerwise', default=True, type=lambda x: bool(int(x)))
    parser.add_argument('--clip_init_val', default=2.5, type=float)
    parser.add_argument('--clip_lr', default=1e-4, type=float)
    parser.add_argument('--clip_wd', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)

    args = parser.parse_args()
    args.do_lower_case = True
    args.do_eval = True

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'record_%s.log' % args.job_id)
    init_logging(log_path)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info('device: %s', device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)

    if args.batch_size is None:
        args.batch_size = default_params[task_name]['batch_size']
    if args.max_seq_length is None:
        args.max_seq_length = default_params[task_name]['max_seq_length']

    mac_types = parse_mac_types(args.mac_types)
    x_levels = parse_sweep(args.x_sweep)
    stats = load_stats(args.stats_file, mac_types)

    print_args(vars(args))
    logging.info('Transform mode: %s', args.transform_mode)
    logging.info('Transform MAC types: %s', mac_types)
    logging.info('x sweep: %s', x_levels)
    for t in mac_types:
        logging.info('  stats[%s]: mean=%.4f std=%.4f', t, stats[t]['mean'], stats[t]['std'])

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.student_model)
    config.num_labels = num_labels

    student_config = copy.deepcopy(config)
    student_config.weight_bits = args.weight_bits
    student_config.input_bits = args.input_bits
    student_config.weight_quant_method = args.weight_quant_method
    student_config.input_quant_method = args.input_quant_method
    student_config.clip_init_val = args.clip_init_val
    student_config.learnable_scaling = args.learnable_scaling
    student_config.sym_quant_qkvo = args.sym_quant_qkvo
    student_config.sym_quant_ffn_attn = args.sym_quant_ffn_attn
    student_config.embed_layerwise = args.embed_layerwise
    student_config.weight_layerwise = args.weight_layerwise
    student_config.input_layerwise = args.input_layerwise
    student_config.hidden_act = args.ACT2FN

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    mm_eval_dataloader = None
    mm_eval_labels = None
    if task_name == 'mnli':
        mm_processor = processors['mnli-mm']()
        mm_eval_examples = mm_processor.get_dev_examples(args.data_dir)
        mm_eval_features = convert_examples_to_features(
            mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)
        mm_eval_sampler = SequentialSampler(mm_eval_data)
        mm_eval_dataloader = DataLoader(
            mm_eval_data, sampler=mm_eval_sampler, batch_size=args.batch_size)

    student_model = BertForSequenceClassification.from_pretrained(
        args.student_model, config=student_config)
    student_model.to(device)
    student_model.eval()

    learner = KDLearner(args, device, student_model, teacher_model=None, num_train_optimization_steps=0)

    csv_path = os.path.join(args.output_dir, 'transform_sweep_results.csv')
    fieldnames = ['x', 'transform_mode', 'mac_types', 'acc', 'acc_mm', 'eval_loss']
    rows = []

    # Make sure no Gaussian noise is ever applied during these runs.
    MacNoiseConfig.reset()

    for x in x_levels:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)

        MacTransform.configure(args.transform_mode, x, mac_types, stats, enabled=True)
        logging.info('===== Evaluating %s x=%.4f =====', args.transform_mode, x)

        try:
            result, mm_acc = evaluate_once(
                learner, task_name, output_mode, eval_labels, num_labels,
                eval_dataloader, mm_eval_dataloader, mm_eval_labels)
        except Exception:
            logging.exception('Evaluation failed at x=%s', x)
            raise

        row = {
            'x': x,
            'transform_mode': args.transform_mode,
            'mac_types': ','.join(mac_types),
            'acc': result.get('acc', ''),
            'eval_loss': result.get('eval_loss', ''),
        }
        if task_name == 'mnli':
            row['acc_mm'] = mm_acc if mm_acc is not None else ''
        rows.append(row)

        logging.info('x=%s acc=%s eval_loss=%s', x, row['acc'], row['eval_loss'])
        if task_name == 'mnli':
            logging.info('x=%s acc_mm=%s', x, row.get('acc_mm', ''))

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    MacTransform.reset()

    logging.info('Wrote results to %s', csv_path)
    return 0


if __name__ == '__main__':
    main()
