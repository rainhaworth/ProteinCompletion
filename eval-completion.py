# eval by completing partial proteins
import os
import time
import random
import numpy as np
import argparse

import torch

from tokenizers import Tokenizer
from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.data import make_gen_from_ext
from utils.mask import idx_to_segments
from utils.utils import print_time, set_env, set_seed, load_compat, create_tokenizer_custom

# import generation step function
from generate import gen_step, make_inference_mask, greedy_sample, nucleus_sample

from tqdm import tqdm


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY' # restrict generation to 20 standard amino acids

def cross_entropy_2way(seq, logits):
    half_sz = logits.size(-1) // 2
    p_logits = logits[1:,:half_sz]
    n_logits = logits[:-1,half_sz:]
    p_toks = seq[0,:-1]
    n_toks = seq[0,1:]

    ce = [torch.nn.functional.cross_entropy(p_logits, p_toks), torch.nn.functional.cross_entropy(n_logits, n_toks)]
    #ce /= 2
    ce = np.array([x.numpy(force=True) for x in ce])
    return ce

def seq_to_ce(seq, model, tokenizer, device):
    idxs = list(range(len(seq)))
    seq = tokenizer.encode(seq).ids
    seq = torch.tensor(seq).to(device)
    seq = seq[None,:]

    mask = make_inference_mask(seq.size(1), idxs, device, seq.size(1))
    logits = model(seq, attention_mask=mask).logits
    logits = torch.squeeze(logits, 0)

    ce = cross_entropy_2way(seq, logits)
    return ce



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/model.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data', type=str, default='./data/uniprot_sprot.fasta')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--rep-window', type=int, default=4)
    parser.add_argument('--rep-penalty', type=float, default=1.2)
    parser.add_argument('--sample', choices=['nucleus', 'greedy'], default='nucleus')
    parser.add_argument('--max-window', type=int, default=-1)
    parser.add_argument('--config', type=str, default='./config-medium.json')
    parser.add_argument('--model_type', choices=['bidirectional','esmlike'], default='bidirectional')
    args = parser.parse_args()

    if args.model_type == 'bidirectional':
        model_class = BidirectionalCausalLM
    else:
        model_class = ESMlikeLM
    
    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # load everything

    with print_time('loading model'):
        model = load_compat(model_class, args.config, device, args.weights, training=False)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

        # get valid token IDs; does not work with proper BPE
        # this excludes terminals, which are handled later
        valid_ids = tokenizer.encode(VALID_AAS).ids
        invalid_ids = [x for x in range(32) if x not in valid_ids]

    with print_time('loading datasets'):
        dataset = make_gen_from_ext(args.data)

    # run eval
    
    model.eval()

    keep_fracs = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    with print_time('evaluating'):
        n = 0
        prev_seq = None
        ppls = []
        for seq, _ in dataset:
            if seq == prev_seq: continue
            if len(seq) < 100 or len(seq) > 5000: continue
            else: prev_seq = seq
            print('seq:', seq)

            init_seq = seq

            # 2 problem settings: contiguous + fragmented subseq
            for contiguous in [False,True]:
                for keep_frac in keep_fracs:
                    # get subseq
                    keep_sz = max(1, int(keep_frac * len(seq)))
                    
                    if contiguous:
                        keep_start = np.random.randint(0, len(seq)-keep_sz+1)
                        keep_idx = np.arange(keep_start, keep_start+keep_sz)
                    else:
                        keep_idx = np.sort(np.random.choice(range(len(seq)), keep_sz, replace=False))
                    
                    # make tensors
                    seq = init_seq[keep_idx[0]:keep_idx[-1]+1]
                    seq = tokenizer.encode(seq).ids
                    seq = [BOS_ID] + seq + [EOS_ID]
                    seqlen = len(seq)
                    seq = torch.tensor(seq).to(device)
                    seq = seq[None,:]
                    idxs = torch.tensor(keep_idx - keep_idx[0])
                    
                    # generate
                    # TODO: constrain to fill in the middle if non contiguous
                    gen_steps = len(seq) - keep_sz
                    for _ in range(gen_steps):
                        # generate next token if possible
                        new_token, new_pos = gen_step(model, seq, idxs, device, invalid_ids)
                        if new_token == None: break

                        # update seq and idxs
                        new_token = new_token[None,None]
                        if new_pos == -1:
                            # prepend, shift all indices up
                            seq = torch.cat([new_token, seq], dim=-1)
                            idxs = torch.cat([new_pos[None], idxs])
                            idxs += 1
                        elif new_pos == seq.size(1):
                            # append
                            seq = torch.cat([seq, new_token], dim=-1)
                            idxs = torch.cat([idxs, new_pos[None]])
                        else:
                            # insert
                            seq[:,new_pos] = new_token
                            idxs = torch.cat([idxs, new_pos[None]]).sort()[0]
                            idxs = idxs.sort()[0]

                    print('generated {:.2f}%, contiguous = {}, CE {}: {}'.format(
                        (1-keep_frac)*100,
                        contiguous,
                        seq_to_ce(seq, model, tokenizer, device),
                        tokenizer.decode(seq.squeeze().numpy(force=True))
                        )
                    )

if __name__ == '__main__':
    main()
    print('done.')
