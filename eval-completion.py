# eval by completing partial proteins
import numpy as np
import argparse
import torch

from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.data import make_gen_from_ext
from utils.utils import print_time, set_env, set_seed, load_compat, create_tokenizer_custom
from utils.generation import gen_step_bidirectional, gen_step_esmlike, make_inference_mask


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY' # restrict generation to 20 standard amino acids

def cross_entropy_2way(logits, seq):
    half_sz = logits.size(-1) // 2
    p_logits = logits[1:,:half_sz]
    n_logits = logits[:-1,half_sz:]
    p_toks = seq[:-1]
    n_toks = seq[1:]

    ce = [torch.nn.functional.cross_entropy(p_logits, p_toks), torch.nn.functional.cross_entropy(n_logits, n_toks)]
    #ce /= 2
    ce = np.array([x.numpy(force=True) for x in ce])
    return ce

# assume tokenized tensor seq
def seq_to_ce(seq : torch.Tensor, model, device, ce_fn=cross_entropy_2way):
    idxs = list(range(seq.size(1)))

    mask = make_inference_mask(seq.size(1), idxs, device, seq.size(1))
    logits = model(seq, attention_mask=mask)
    ce = ce_fn(logits.squeeze(0), seq.squeeze(0))
    return ce

# compute shannon entropy by character frequencies
def seq_entropy(seq : str, ignore_idxs):
    idxs = set(ignore_idxs)
    freqs = {c: 0 for c in VALID_AAS}
    for i, c in enumerate(seq):
        if i in idxs: continue
        freqs[c] += 1
    freqs = np.array(list(freqs.values()), dtype=float)
    freqs = freqs[freqs != 0]
    freqs /= np.sum(freqs)
    return -np.sum(freqs * np.log2(freqs))


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
        gen_step = gen_step_bidirectional
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
        for seq, _ in dataset:
            if seq == prev_seq: continue
            if len(seq) < 100 or len(seq) > 5000: continue

            print('seq:', seq)
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
                    seq = torch.tensor(seq).to(device)
                    seq = seq[None,:]
                    idxs = torch.tensor(keep_idx).to(device)
                    
                    # generate
                    # TODO: constrain to fill in the middle if non contiguous
                    gen_steps = len(prev_seq) - keep_sz
                    for gs in range(gen_steps):
                        # generate next token
                        new_token, new_pos = gen_step(model, seq, idxs, device, invalid_ids, predict_terminals=False)
                        if new_token == None:
                            print('generation failed on step', gs)
                            print('seq', seq)
                            print('idxs', idxs)
                            break

                        # update seq and idxs
                        seq[:,new_pos] = new_token[None,None]
                        idxs = torch.cat([idxs, new_pos[None]]).sort()[0]

                    seq_str = tokenizer.decode(seq.squeeze().numpy(force=True))
                    print('generated {:.2f}%, contiguous = {}, PPL {}, SE {}: {}'.format(
                        (1-keep_frac)*100,
                        contiguous,
                        2 ** seq_to_ce(seq, model, device, ce_fn),
                        seq_entropy(seq_str, keep_idx),
                        tokenizer.decode(seq.squeeze().numpy(force=True))
                        )
                    )
            n += 1
            if n > 10: break
            print()

if __name__ == '__main__':
    main()
    print('done.')
