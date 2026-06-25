# coding=utf-8
"""Collect integer binary-MAC output distributions from a trained checkpoint.

Runs a forward pass over a few eval batches (no noise) and accumulates, per
selected MAC type, a histogram of the integer MAC outputs. Both per-layer and
pooled-over-all-layers histograms are written to a long-format CSV that the
plotting script consumes.
"""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import csv
import logging
import os
import random

import torch
from torch.utils.data import DataLoader, SequentialSampler

from binary_mac_noise import MacOutputCollector, VALID_MAC_TYPES, register_mac_sites
from helper import init_logging, print_args
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


def parse_mac_types(mac_types_str):
    mac_types = [x.strip() for x in mac_types_str.split(',') if x.strip()]
    if not mac_types:
        raise ValueError('--mac_types must list at least one MAC type')
    unknown = set(mac_types) - set(VALID_MAC_TYPES)
    if unknown:
        raise ValueError('Unknown MAC types: %s. Valid: %s'
                         % (sorted(unknown), sorted(VALID_MAC_TYPES)))
    return mac_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--student_model', required=True, type=str,
                        help='Path to trained checkpoint dir (e.g. outputs/MNLI/W1A1/kd_joint)')
    parser.add_argument('--vocab_dir', required=True, type=str)
    parser.add_argument('--task_name', default='MNLI', type=str)
    parser.add_argument('--output_dir', default='./mac_dist_results', type=str)
    parser.add_argument('--job_id', default='mac_dist', type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--max_seq_length', default=None, type=int)
    parser.add_argument('--num_batches', default=8, type=int,
                        help='Number of eval batches to run (<=0 means whole dev set)')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mac_types', required=True, type=str,
                        help='Comma-separated MAC types: %s' % ', '.join(sorted(VALID_MAC_TYPES)))

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

    args = parser.parse_args()
    args.do_lower_case = True

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
    print_args(vars(args))
    logging.info('Collecting MAC types: %s', mac_types)
    logging.info('num_batches: %s', args.num_batches)

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
    eval_data, _ = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    student_model = BertForSequenceClassification.from_pretrained(
        args.student_model, config=student_config)
    student_model.to(device)
    student_model.eval()

    register_mac_sites(student_model)
    MacOutputCollector.configure(mac_types, enabled=True)

    n_seen = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            if args.num_batches > 0 and n_seen >= args.num_batches:
                break
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, _, _ = batch
            student_model(input_ids, segment_ids, input_mask)
            n_seen += 1

    logging.info('Ran %d batches (batch_size=%d)', n_seen, args.batch_size)

    rows = MacOutputCollector.export_rows()
    MacOutputCollector.reset()

    csv_path = os.path.join(args.output_dir, 'mac_distribution.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['mac_type', 'layer_idx', 'value', 'count'])
        writer.writerows(rows)

    logging.info('Wrote %d histogram rows to %s', len(rows), csv_path)
    return 0


if __name__ == '__main__':
    main()
