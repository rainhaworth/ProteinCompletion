"""Evaluate ATP adjacent-token scoring or ESM masked-residue recovery."""

import argparse
import csv
import os

import torch

from utils.data import make_gen_from_ext
from utils.evaluation import normalize_model_type, score_atp_logits, score_esm_logits
from utils.generation import make_inference_mask, make_mlm_input
from utils.metrics import cross_entropy_to_perplexity
from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.utils import (
    create_tokenizer_custom,
    load_model_checkpoint,
    print_time,
    set_env,
    set_seed,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Score ATP adjacent-token predictions or ESM masked residues.'
    )
    parser.add_argument('--weights', required=True, help='Training checkpoint or serialized model')
    parser.add_argument('--data', required=True, help='Input FASTA or TSV')
    parser.add_argument('--output', default='', help='Optional per-sequence TSV output')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', default='./config-medium.json')
    parser.add_argument('--tokenizer', default='./tokenizer-uniref.json')
    parser.add_argument(
        '--model_type',
        choices=['atp', 'esm', 'bidirectional', 'esmlike'],
        default='atp',
    )
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--min-length', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=1000)
    parser.add_argument(
        '--mask-fraction',
        type=float,
        default=0.15,
        help='Fraction of positions hidden when evaluating ESM.',
    )
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument(
        '--rng-deterministic',
        default=True,
        type=lambda value: str(value).lower() == 'true',
    )
    return parser


def tokenize_sequence(sequence, tokenizer, device):
    token_ids = tokenizer.encode(sequence).ids
    if not token_ids:
        raise ValueError('Tokenizer produced an empty sequence')
    if len(token_ids) != len(sequence):
        raise ValueError('Evaluation requires one token per residue')
    return torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]


def score_sequence(model, seq, model_type, device, mask_fraction=0.15, mask_id=None):
    with torch.inference_mode():
        if model_type == 'atp':
            known = torch.arange(seq.size(1), device=device)
            attention_mask = make_inference_mask(
                seq.size(1), known, device, seq.size(1)
            )
            return score_atp_logits(
                seq, model(seq, attention_mask=attention_mask)
            )

        target_count = max(1, min(seq.size(1), round(seq.size(1) * mask_fraction)))
        target_indices = torch.randperm(seq.size(1), device=device)[:target_count]
        if mask_id is None:
            raise ValueError('ESM scoring requires the tokenizer mask ID')
        known = torch.ones(seq.size(1), dtype=torch.bool, device=device)
        known[target_indices] = False
        model_input = make_mlm_input(seq, torch.nonzero(known).squeeze(-1), mask_id)
        logits = model(model_input, attention_mask=None)
        result = score_esm_logits(seq, logits, target_indices)
        result['target_indices'] = target_indices
        return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError('Invalid sequence-length bounds')
    if not 0 < args.mask_fraction <= 1:
        raise ValueError('--mask-fraction must be greater than 0 and at most 1')

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        print('CUDA is unavailable; using CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    model_type = normalize_model_type(args.model_type)
    model_class = BidirectionalCausalLM if model_type == 'atp' else ESMlikeLM

    with print_time('loading model'):
        model = load_model_checkpoint(
            model_class, args.config, device, args.weights
        ).eval()
    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(args.tokenizer)
        mask_id = tokenizer.token_to_id('<mask>') if model_type == 'esm' else None
        if model_type == 'esm' and mask_id is None:
            raise ValueError('Tokenizer does not define a <mask> token')
    with print_time('loading dataset'):
        dataset = make_gen_from_ext(args.data)

    rows = []
    total_cross_entropy = 0.0
    total_correct = 0.0
    total_targets = 0
    previous_sequence = None

    with print_time('evaluating'):
        for sequence, _ in dataset:
            if sequence == previous_sequence:
                continue
            previous_sequence = sequence
            if not args.min_length <= len(sequence) <= args.max_length:
                continue

            seq = tokenize_sequence(sequence, tokenizer, device)
            if seq.size(1) > model.config.n_ctx:
                continue
            result = score_sequence(
                model, seq, model_type, device, args.mask_fraction, mask_id
            )
            cross_entropy = float(result['cross_entropy'].detach().cpu())
            accuracy = float(result['accuracy'].detach().cpu())
            targets = int(result['targets'])
            total_cross_entropy += cross_entropy * targets
            total_correct += accuracy * targets
            total_targets += targets

            row = {
                'record_id': len(rows) + 1,
                'length': seq.size(1),
                'targets': targets,
                'cross_entropy': cross_entropy,
                'perplexity': float(cross_entropy_to_perplexity(cross_entropy)),
                'accuracy': accuracy,
                'sequence': sequence,
            }
            if model_type == 'esm':
                row['target_indices'] = ' '.join(
                    map(str, result['target_indices'].detach().cpu().tolist())
                )
            else:
                row['previous_cross_entropy'] = float(
                    result['previous_cross_entropy'].detach().cpu()
                )
                row['next_cross_entropy'] = float(
                    result['next_cross_entropy'].detach().cpu()
                )
            rows.append(row)
            print(
                f"record {row['record_id']} length={row['length']} "
                f"CE={cross_entropy:.5f} accuracy={accuracy:.5f}"
            )
            if len(rows) >= args.max_samples:
                break

    if not rows:
        raise ValueError('No sequences satisfied the evaluation filters')

    mean_cross_entropy = total_cross_entropy / total_targets
    mean_accuracy = total_correct / total_targets
    print(f'token-weighted cross-entropy: {mean_cross_entropy:.5f}')
    print(
        'token-weighted perplexity:',
        f'{float(cross_entropy_to_perplexity(mean_cross_entropy)):.5f}',
    )
    print(f'token-weighted accuracy: {mean_accuracy:.5f}')

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(output_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        print('saved to', output_path)


if __name__ == '__main__':
    main()
