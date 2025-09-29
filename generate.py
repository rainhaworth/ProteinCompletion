# generate complete proteins from fragments
import numpy as np
import argparse
import torch

from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM

# import custom dataset
from utils.data import ProteinBindingOnlyData
from utils.generation import make_inference_mask, gen_step_bidirectional, gen_step_esmlike, greedy_sample, nucleus_sample
from utils.utils import print_time, set_seed, set_env, create_tokenizer_custom, load_compat

from tqdm import tqdm


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VALID_AAS = 'ACDEFGHIKLMNPQRSTVWY' # restrict generation to 20 standard amino acids


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
    parser.add_argument('--config', type=str, default='./config-medium.json')
    parser.add_argument('--model_type', choices=['bidirectional','esmlike'], default='bidirectional')
    args = parser.parse_args()

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    if args.model_type == 'bidirectional':
        model_class = BidirectionalCausalLM
        gen_step = gen_step_bidirectional
        keep_len = False
    else:
        model_class = ESMlikeLM
        gen_step = gen_step_esmlike
        keep_len = True

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    with print_time('loading model'):
        model = load_compat(model_class, args.config, device, args.weights, training=False)

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

        # get valid token IDs; does not work with proper BPE
        # this excludes terminals, which are handled later
        valid_ids = tokenizer.encode(VALID_AAS).ids
        invalid_ids = [x for x in range(32) if x not in valid_ids]

    # load dataset

    with print_time('loading datasets'):
        dataset = ProteinBindingOnlyData(args.data, tokenizer, max_dim=2048, max_samples=15, keep_len=keep_len)
        dataloader = torch.utils.data.DataLoader(dataset)

    # (4) generate

    max_steps = args.max_steps
    rw = args.rep_window
    rp = args.rep_penalty

    # sample_fn input: logits, output: (index along logits dim=0, token ID)
    if args.sample == 'nucleus':
        sample_fn = nucleus_sample
    else:
        sample_fn = greedy_sample

    model.eval()

    with print_time('generating'):
        ppls = []
        for seq, idxs in dataloader:
            print('binding site:\t', tokenizer.decode(seq.squeeze(0).numpy()))
            print('idxs:\t\t', idxs.squeeze().numpy().tolist())
            # put everything on the GPU
            seq = seq.to(device)
            idxs = idxs.to(device)

            idxs = idxs.squeeze(0)

            for _ in tqdm(range(max_steps)):
                # generate next token if possible
                # TODO: custom gen_step for ESMlikeLM
                new_token, new_pos = gen_step(model, seq, idxs, device, invalid_ids, rp, rw, sample_fn)
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

            print('generated:\t', tokenizer.decode(seq.squeeze().numpy(force=True)))

            # compute CE; if bidirectional take mean across next and prev positions
            mask = make_inference_mask(seq.size(1), idxs, device, seq.size(1))
            logits = model(seq, attention_mask=mask)
            logits = torch.squeeze(logits, 0)

            if args.model_type == 'bidirectional':
                half_sz = logits.size(-1) // 2
                p_logits = logits[1:,:half_sz]
                n_logits = logits[:-1,half_sz:]
                p_toks = seq[0,:-1]
                n_toks = seq[0,1:]

                ce = torch.nn.functional.cross_entropy(p_logits, p_toks) + torch.nn.functional.cross_entropy(n_logits, n_toks)
                ce /= 2
                ce = ce.numpy(force=True)
            else:
                ce = torch.nn.functional.cross_entropy(logits, seq[0])
                ce = ce.numpy(force=True)

            print('CE:\t', ce)
            print('PPL:\t', 2 ** ce, end='\n\n')
            ppls.append(2**ce)
        print('mean PPL:', np.mean(ppls))


if __name__ == '__main__':
    main()
    print('done.')
