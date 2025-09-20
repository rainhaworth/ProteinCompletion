# simple PPL + accuracy eval
import numpy as np
import argparse
import json

import torch
from utils.model_bidirectional import BidirectionalCausalLM
from utils.config import BaseConfig

# import custom dataset
from utils.data import make_gen_from_ext
from utils.mask import idx_to_segments
from utils.utils import print_time, set_seed, set_env, create_tokenizer_custom

# import generation step function
from generate import gen_step, make_inference_mask, greedy_sample, nucleus_sample

from tqdm import tqdm


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MAX_ID = 29


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
    parser.add_argument('--acc', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--config', type=str, default='./config-medium.json')
    args = parser.parse_args()


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
        model = torch.load(args.weights, weights_only=False)
        # if dict, expect config arg to be provided
        if type(model) is dict:
            dt = model
            with open(args.config, 'r') as f:
                cj = json.load(f)
            config = BaseConfig(
                cj['vocab_size'],
                cj['n_positions'],
                cj['n_ctx'],
                cj['n_embd'],
                cj['n_layer'],
                cj['n_head'],
                resid_pdrop=cj['resid_pdrop'],
                embd_pdrop=cj['embd_pdrop'],
                attn_pdrop=cj['embd_pdrop'],
                use_cache=False,
                bos_token_id=1,
                eos_token_id=2
            )
            model = BidirectionalCausalLM(config)
            model.load_state_dict(dt['model_state'])
            model.to(device)

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    # load dataset

    with print_time('loading datasets'):
        dataset = make_gen_from_ext(args.data)

    # run eval

    model.eval()

    with print_time('evaluating'):
        n = 0
        prev_seq = None
        ppls = []
        i = 0
        for seq, _ in dataset:
            i += 1
            #if i < 100000: continue
            if seq == prev_seq: continue
            if len(seq) > 500 or len(seq) > 5000: continue
            else: prev_seq = seq
            print('seq:', seq)

            ce = seq_to_ce(seq, model, tokenizer, device)

            print('CE:\t', ce)
            print('PPL:\t', 2 ** ce)
            ppls.append(2**ce)

            # TODO: soft acc

            if args.acc:
                # prepare for acc computation
                seq = tokenizer.encode(seq).ids
                seq = [BOS_ID] + seq + [EOS_ID]
                seqlen = len(seq)
                seq = torch.tensor(seq).to(device)
                seq = seq[None,:]
                idxs = torch.tensor(list(range(seqlen)))
                
                #print(seq)

                # acc starting from random subseq of length 5
                w_init = 5
                w = w_init
                sub_start = np.random.randint(seqlen-w-1)
                subseq = seq[:,sub_start:sub_start+w]
                acc_r = 0.0
                for _ in tqdm(range(w, seqlen)):
                    # normal generation
                    new_token, new_pos = gen_step(model, subseq, idxs[:w], device)
                    
                    # update subseq
                    if new_pos == -1:
                        sub_start -= 1
                        tgt = seq[0, sub_start]
                        #print(seq[0, sub_start-w_init:sub_start+w_init], new_token)
                    else:
                        tgt = seq[0, sub_start + w]
                        #print(seq[0, sub_start+w-w_init:sub_start+w+w_init], new_token)
                    w += 1
                    
                    #print(i, w, sub_start, w+sub_start, new_token, tgt, new_pos)
                    subseq = seq[:,sub_start:sub_start+w]
                    #print(subseq)

                    acc_r += (new_token == tgt)#.numpy(force=True)
                
                print('hard acc r:', acc_r.numpy(force=True) / (seqlen-w_init))

                # forward only acc
                acc_f = 0.0
                for i in tqdm(range(1,seqlen-0)):
                    # get predictions
                    subseq = seq[:,:i]
                    logits, _ = gen_step(model, subseq, idxs[:i], device, return_logits=True)

                    # use raw prediction weight for next token as accuracy
                    acc_f += (torch.argmax(logits[-1]) == seq[0,i]).numpy(force=True)
                
                print('hard acc f:', acc_f / seqlen)
                
                # backward only acc
                acc_b = 0.0
                for i in tqdm(range(1,seqlen-0)):
                    # get predictions
                    subseq = seq[:,-i:]
                    logits, _ = gen_step(model, subseq, idxs[:i], device, return_logits=True)

                    # use raw prediction weight for next token as accuracy
                    acc_b += (torch.argmax(logits[0]) == seq[0,-i-1]).numpy(force=True)
                
                print('hard acc b:', acc_b / seqlen)

            print()

            n += 1
            if n > 100: break
        print('mean PPL:', np.mean(np.concatenate(ppls)))
        
        seq = '1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRMSAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRTAARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAVYRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2'
        print('baseline:', seq)

        ce = seq_to_ce(seq, model, tokenizer, device)

        print('CE:\t', ce)
        print('PPL:\t', 2 ** ce, end='\n\n')

if __name__ == '__main__':
    main()
    print('done.')
