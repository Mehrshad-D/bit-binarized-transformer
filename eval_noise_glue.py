# coding=utf-8
"""Evaluate GLUE checkpoints with In-DRAM binary MAC noise simulation."""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import csv
import logging
import os
import random

import torch
from torch.utils.data import DataLoader, SequentialSampler

from binary_mac_noise import MacNoiseConfig, VALID_MAC_TYPES
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
        raise ValueError('--noise_mac_types must list at least one MAC type')
    return mac_types


def evaluate_once(learner, task_name, output_mode, eval_labels, num_labels,
                  eval_dataloader, eval_examples, mm_eval_dataloader, mm_eval_labels):
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
    parser.add_argument('--output_dir', default='./noise_results', type=str)
    parser.add_argument('--job_id', default='noise_eval', type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--max_seq_length', default=None, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_sweep', default='0,1,2,5,10', type=str,
                        help='Comma-separated noise percentages (x%% of MAC size, 3-sigma rule)')
    parser.add_argument('--noise_mac_types', required=True, type=str,
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

    mac_types = parse_mac_types(args.noise_mac_types)
    noise_levels = parse_sweep(args.noise_sweep)

    print_args(vars(args))
    logging.info('Noise MAC types: %s', mac_types)
    logging.info('Noise sweep (%%): %s', noise_levels)

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

    csv_path = os.path.join(args.output_dir, 'noise_sweep_results.csv')
    fieldnames = ['noise_pct', 'noise_mac_types', 'acc', 'acc_mm', 'eval_loss']
    rows = []

    for noise_pct in noise_levels:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)

        MacNoiseConfig.configure(noise_pct, mac_types, enabled=True)
        logging.info('===== Evaluating noise_pct=%.4f =====', noise_pct)

        try:
            result, mm_acc = evaluate_once(
                learner, task_name, output_mode, eval_labels, num_labels,
                eval_dataloader, eval_examples, mm_eval_dataloader, mm_eval_labels)
        except Exception:
            logging.exception('Evaluation failed at noise_pct=%s', noise_pct)
            raise

        row = {
            'noise_pct': noise_pct,
            'noise_mac_types': ','.join(mac_types),
            'acc': result.get('acc', ''),
            'eval_loss': result.get('eval_loss', ''),
        }
        if task_name == 'mnli':
            row['acc_mm'] = mm_acc if mm_acc is not None else ''
        rows.append(row)

        logging.info('noise_pct=%s acc=%s eval_loss=%s', noise_pct, row['acc'], row['eval_loss'])
        if task_name == 'mnli':
            logging.info('noise_pct=%s acc_mm=%s', noise_pct, row.get('acc_mm', ''))

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    MacNoiseConfig.reset()

    logging.info('Wrote results to %s', csv_path)
    return 0


if __name__ == '__main__':
    main()
