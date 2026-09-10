"""Generate protein completions from visible sequence motifs."""

import argparse
import csv
import os

import numpy as np
import torch

from utils.data import ProteinBindingOnlyData
from utils.evaluation import normalize_model_type, score_atp_logits, score_esm_logits
from utils.generation import (
    decode_token_ids,
    decode_visible_token_ids,
    gen_step_atp,
    gen_step_esmlike,
    make_inference_mask,
    make_sample_fn,
    resolve_terminal_ids,
    update_generation_state,
)
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


VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY'


def build_parser():
    parser = argparse.ArgumentParser(
        description='Complete protein sequences from visible motif positions.'
    )
    parser.add_argument('--weights', required=True, help='Training checkpoint or serialized model')
    parser.add_argument('--data', required=True, help='Input FASTA or binding-site TSV')
    parser.add_argument('--output', default='', help='Optional TSV output path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', default='./config-medium.json')
    parser.add_argument('--tokenizer', default='./tokenizer-uniref.json')
    parser.add_argument(
        '--model_type',
        choices=['atp', 'esm', 'bidirectional', 'esmlike'],
        default='atp',
    )
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument(
        '--rng-deterministic',
        default=True,
        type=lambda value: str(value).lower() == 'true',
    )
    parser.add_argument('--sample', choices=['nucleus', 'greedy'], default='nucleus')
    parser.add_argument('--p', type=float, default=0.95, help='Nucleus probability threshold')
    parser.add_argument('--max-samples', type=int, default=15)
    parser.add_argument(
        '--motif-dropout',
        type=float,
        default=0.0,
        help='Optional dropout applied to annotated TSV motif positions.',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=0,
        help='Maximum generated residues per sample. Zero uses the model context limit.',
    )
    return parser


def fully_visible_reconstruction_score(model, seq, model_type, device):
    """Score token reconstruction with every sequence position visible."""
    with torch.inference_mode():
        if model_type == 'atp':
            known = torch.arange(seq.size(1), device=device)
            attention_mask = make_inference_mask(
                seq.size(1), known, device, seq.size(1)
            )
            result = score_atp_logits(seq, model(seq, attention_mask=attention_mask))
        else:
            result = score_esm_logits(seq, model(seq, attention_mask=None))
    cross_entropy = float(result['cross_entropy'].detach().cpu())
    accuracy = float(result['accuracy'].detach().cpu())
    return cross_entropy, float(cross_entropy_to_perplexity(cross_entropy)), accuracy


def prepare_generation_output(
    seq,
    visible_indices,
    motif_indices,
    tokenizer,
    complete,
    bos_id,
    eos_id,
):
    """Remove terminal tokens and preserve the original motif coordinates."""
    token_ids = seq.squeeze(0)
    start = int(token_ids.numel() > 0 and token_ids[0].item() == bos_id)
    end = token_ids.numel() - int(
        token_ids.numel() > start and token_ids[-1].item() == eos_id
    )
    output_ids = token_ids[start:end]

    def shift_and_filter(indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=seq.device).reshape(-1)
        indices = indices[(indices >= start) & (indices < end)] - start
        return torch.unique(indices, sorted=True)

    output_visible = shift_and_filter(visible_indices)
    output_motif = shift_and_filter(motif_indices)
    if complete:
        sequence = decode_token_ids(tokenizer, output_ids)
    else:
        sequence = decode_visible_token_ids(
            tokenizer, output_ids, output_visible
        )
    return sequence, output_motif, output_visible


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    if args.max_steps < 0:
        raise ValueError('--max-steps cannot be negative')
    if not 0 <= args.motif_dropout <= 1:
        raise ValueError('--motif-dropout must be between 0 and 1')

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        print('CUDA is unavailable; using CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    model_type = normalize_model_type(args.model_type)

    if model_type == 'atp':
        model_class = BidirectionalCausalLM
        generation_step = gen_step_atp
        keep_length = False
    else:
        model_class = ESMlikeLM
        generation_step = gen_step_esmlike
        keep_length = True

    with print_time('loading model'):
        model = load_model_checkpoint(
            model_class, args.config, device, args.weights
        ).eval()
    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file=args.tokenizer)
        bos_id, eos_id = resolve_terminal_ids(tokenizer)
        mask_id = None
        if model_type == 'esm':
            mask_id = tokenizer.token_to_id('<mask>')
            if mask_id is None:
                raise ValueError('Tokenizer does not define a <mask> token')
        valid_ids = set(tokenizer.encode(VALID_AAS).ids)
        invalid_ids = [
            token_id
            for token_id in range(model.config.vocab_size)
            if token_id not in valid_ids
        ]
    with print_time('loading dataset'):
        dataset = ProteinBindingOnlyData(
            args.data,
            tokenizer,
            max_dim=model.config.n_ctx,
            max_samples=args.max_samples,
            keep_len=keep_length,
            motif_dropout=args.motif_dropout,
        )
    if not dataset:
        raise ValueError(f'No sequences were loaded from {args.data}')

    sample_fn = make_sample_fn(args.sample, args.p)
    step_limit = args.max_steps or model.config.n_ctx
    rows = []

    with print_time('generating'), torch.inference_mode():
        for record_id, (seq, idxs) in enumerate(dataset, start=1):
            seq = seq[None, :].to(device)
            idxs = torch.as_tensor(idxs, dtype=torch.long, device=device).reshape(-1)
            motif_indices = idxs.clone()

            for _ in range(step_limit):
                new_token, new_pos = generation_step(
                    model,
                    seq,
                    idxs,
                    device,
                    invalid_ids,
                    sample_fn=sample_fn,
                    **(
                        {'mask_id': mask_id}
                        if model_type == 'esm'
                        else {'bos_id': bos_id, 'eos_id': eos_id}
                    ),
                )
                if new_token is None:
                    break
                if int(torch.as_tensor(new_pos).item()) == -1:
                    motif_indices = motif_indices + 1
                seq, idxs = update_generation_state(
                    seq, idxs, new_token, new_pos
                )
                if seq.size(1) >= model.config.n_ctx and model_type == 'atp':
                    break

            all_positions_visible = idxs.numel() == seq.size(1)
            has_terminals = (
                seq.size(1) >= 2
                and seq[0, 0].item() == bos_id
                and seq[0, -1].item() == eos_id
            )
            complete = (
                all_positions_visible
                if model_type == 'esm'
                else all_positions_visible and has_terminals
            )
            sequence, output_motif, output_visible = prepare_generation_output(
                seq,
                idxs,
                motif_indices,
                tokenizer,
                complete,
                bos_id,
                eos_id,
            )
            if complete:
                cross_entropy, perplexity, accuracy = fully_visible_reconstruction_score(
                    model, seq, model_type, device
                )
            else:
                cross_entropy = perplexity = accuracy = np.nan

            print(f'record {record_id}')
            print('complete:', complete)
            print('sequence:', sequence)
            if complete:
                print(f'fully-visible reconstruction CE: {cross_entropy:.5f}')
                print(f'fully-visible reconstruction PPL: {perplexity:.5f}')
                print(f'fully-visible reconstruction accuracy: {accuracy:.5f}')
            rows.append(
                {
                    'record_id': record_id,
                    'model_type': model_type,
                    'complete': complete,
                    'known_indices': ' '.join(
                        map(str, output_motif.detach().cpu().tolist())
                    ),
                    'visible_indices': ' '.join(
                        map(str, output_visible.detach().cpu().tolist())
                    ),
                    'fully_visible_reconstruction_cross_entropy': cross_entropy,
                    'fully_visible_reconstruction_perplexity': perplexity,
                    'fully_visible_reconstruction_accuracy': accuracy,
                    'sequence': sequence,
                }
            )

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        print('saved to', output_path)


if __name__ == '__main__':
    main()
