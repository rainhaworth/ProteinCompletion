# eval by completing partial proteins
import numpy as np
import argparse
import torch
import os

from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.data import make_gen_from_ext
from utils.utils import print_time, set_env, set_seed, load_model_checkpoint, create_tokenizer_custom
from utils.generation import (
    decode_token_ids,
    gen_step_esmlike,
    make_inference_mask,
    gen_step_atp,
    make_sample_fn,
    resolve_terminal_ids,
    update_generation_state,
)
from utils.metrics import amino_acid_composition_entropy, cross_entropy_to_perplexity

VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY' # restrict generation to 20 standard amino acids

def cross_entropy_2way(logits, seq):
    if seq.numel() < 2:
        raise ValueError('ATP reconstruction scoring requires at least two residues')
    half_sz = logits.size(-1) // 2
    p_logits = logits[1:,:half_sz]
    n_logits = logits[:-1,half_sz:]
    p_toks = seq[:-1]
    n_toks = seq[1:]

    ce = [torch.nn.functional.cross_entropy(p_logits, p_toks), torch.nn.functional.cross_entropy(n_logits, n_toks)]
    return torch.stack(ce).mean()

# assume tokenized tensor seq
def seq_to_ce(seq : torch.Tensor, model, device, ce_fn=cross_entropy_2way):
    idxs = list(range(seq.size(1)))

    mask = make_inference_mask(seq.size(1), idxs, device, seq.size(1))
    with torch.inference_mode():
        logits = model(seq, attention_mask=mask)
        ce = ce_fn(logits.squeeze(0), seq.squeeze(0))
    return ce

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--data', type=str, default='./data/uniprot_sprot.fasta')
    parser.add_argument('--tokenizer', type=str, default='./tokenizer-uniref.json')
    parser.add_argument('--sample', choices=['nucleus', 'greedy'], default='nucleus')
    parser.add_argument('--config', type=str, default='config-medlarge.json')
    parser.add_argument('--model_type', choices=['atp', 'esm'], default='atp')
    parser.add_argument('--output', default='./out.tsv')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--min-length', type=int, default=100)
    parser.add_argument('--max-length', type=int, default=1000)
    parser.add_argument(
        '--keep-fracs',
        type=float,
        nargs='+',
        default=[0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    )
    args = parser.parse_args(argv)

    if args.max_samples < 0:
        raise ValueError('--max-samples cannot be negative')
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError('Invalid sequence-length bounds')
    if args.model_type == 'atp' and args.min_length < 2:
        raise ValueError('--min-length must be at least 2 for ATP')
    if not args.keep_fracs or any(
        fraction <= 0 or fraction >= 1 for fraction in args.keep_fracs
    ):
        raise ValueError('--keep-fracs values must be between 0 and 1')

    if args.model_type == 'atp':
        model_class = BidirectionalCausalLM
        gen_step = gen_step_atp#gen_step_bidirectional
        ce_fn = cross_entropy_2way
    else:
        model_class = ESMlikeLM
        gen_step = gen_step_esmlike
        ce_fn = torch.nn.functional.cross_entropy
    
    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)

    # load everything

    with print_time('loading model'):
        model = load_model_checkpoint(
            model_class, args.config, device, args.weights
        )

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file=args.tokenizer)
        bos_id, eos_id = resolve_terminal_ids(tokenizer)

        mask_id = None
        if args.model_type == 'esm':
            mask_id = tokenizer.token_to_id('<mask>')
            if mask_id is None:
                raise ValueError('Tokenizer does not define a <mask> token')

        # get valid token IDs; does not work with proper BPE
        # this excludes terminals, which are handled later
        valid_ids = set(tokenizer.encode(VALID_AAS).ids)
        invalid_ids = [
            token_id
            for token_id in range(model.config.vocab_size)
            if token_id not in valid_ids
        ]

    with print_time('loading datasets'):
        dataset = make_gen_from_ext(args.data)

    sample_fn = make_sample_fn(args.sample, args.p)

    # run eval
    
    model.eval()

    keep_fracs = args.keep_fracs

    output_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_parent, exist_ok=True)

    with print_time('evaluating'), open(args.output, 'w') as outf, torch.inference_mode():
        outf.write('generated %\tcontiguous\tfully-visible reconstruction PPL\tAA composition entropy\tidx\tseq\n')
        prev_seq = None
        evaluated_sequences = 0
        
        for seq, _ in dataset:
            if seq == prev_seq: continue
            if len(seq) < args.min_length or len(seq) > args.max_length: continue
            if len(seq) > model.config.n_ctx: continue

            prev_seq = seq

            # 2 problem settings: contiguous + fragmented subseq
            for contiguous in [False,True]:
                for keep_frac in keep_fracs:
                    # get subseq idxs
                    keep_sz = max(1, int(keep_frac * len(prev_seq)))
                    
                    if contiguous:
                        keep_start = np.random.randint(0, len(prev_seq)-keep_sz+1)
                        keep_idx = np.arange(keep_start, keep_start+keep_sz)
                    else:
                        keep_idx = np.sort(np.random.choice(range(len(prev_seq)), keep_sz, replace=False))
                    
                    # make tensors
                    seq = tokenizer.encode(prev_seq).ids
                    if len(seq) != len(prev_seq):
                        raise ValueError(
                            'Completion requires a tokenizer with one token per residue'
                        )
                    seq = torch.tensor(seq).to(device)
                    seq = seq[None,:]
                    idxs = torch.tensor(keep_idx).to(device)
                    
                    # generate
                    # TODO: unit test to confirm no corner cases where some residues are not generated
                    gen_steps = len(prev_seq) - keep_sz
                    for gs in range(gen_steps): # removed tqdm
                        # generate next token
                        gen_kwargs = {'mask_id': mask_id} if args.model_type == 'esm' else {}
                        if args.model_type == 'atp':
                            gen_kwargs.update(bos_id=bos_id, eos_id=eos_id)
                        new_token, new_pos = gen_step(
                            model,
                            seq,
                            idxs,
                            device,
                            invalid_ids,
                            sample_fn=sample_fn,
                            predict_terminals=False,
                            **gen_kwargs,
                        )
                        if new_token is None:
                            raise RuntimeError(
                                f'Generation stopped after {gs} of {gen_steps} steps'
                            )

                        # update seq and idxs
                        seq, idxs = update_generation_state(
                            seq, idxs, new_token, new_pos
                        )

                    if idxs.numel() != seq.size(1):
                        raise RuntimeError(
                            'Generation ended before every hidden position was filled'
                        )

                    seq_str = decode_token_ids(tokenizer, seq.squeeze(0))
                    if len(seq_str) != seq.size(1):
                        raise ValueError(
                            'Decoded sequence length does not match residue indices'
                        )
                    print(seq_str)
                    outf.write('{:.2f}\t{}\t{}\t{:.2f}\t{}\t{}\n'.format(
                        100 * (len(prev_seq) - keep_sz) / len(prev_seq),
                        contiguous,
                        cross_entropy_to_perplexity(seq_to_ce(seq, model, device, ce_fn)),
                        amino_acid_composition_entropy(seq_str, keep_idx),
                        keep_idx,
                        seq_str
                        )
                    )

            evaluated_sequences += 1
            if args.max_samples and evaluated_sequences >= args.max_samples:
                break

    if evaluated_sequences == 0:
        raise ValueError('No sequences satisfied the evaluation filters')

if __name__ == '__main__':
    main()
    print('done.')
