# training script modified from https://github.com/salesforce/progen/blob/main/progen2/sample.py
import os
import argparse
import json
import torch

from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.config import BaseConfig
from utils.data import ProteinBindingData, MaskedProteinData
from utils.utils import print_time, set_seed, set_env, create_tokenizer_custom

from transformers import get_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-medium')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train', type=str, default='./data/uniprot_sprot.fasta')
    parser.add_argument('--eval', type=str, default='')
    parser.add_argument('--save', type=str, default='./weights')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-samples', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=100000)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--model_type', choices=['bidirectional','esmlike'], default='bidirectional')
    args = parser.parse_args()

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    configf = f'./{args.config}.json'
    ckpt = args.ckpt
    bidirectional = (args.model_type == 'bidirectional')

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # load model, parameters, tokenizer

    with print_time('loading parameters'):
        with open(configf, 'r') as f:
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

        if bidirectional:
            model = BidirectionalCausalLM(config)
        else:
            model = ESMlikeLM(config)
        
        if ckpt != '' and os.path.exists(ckpt):
            print('loading from checkpoint')
            states = torch.load(ckpt, map_location='cpu')
            start_step = states['step']
            model.load_state_dict(states['model_state'])
        else:
            print('training from scratch')
            start_step = 0
            states = None
        
        # optimizer ends up on cpu if we don't declare after model.to(device)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        if states is not None: optimizer.load_state_dict(states['optim_state'])


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    # load dataset(s)
    
    # helper function; keep it small and simple for now
    def make_dataloader(dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=args.bsz, shuffle=True)

    with print_time('loading up to ' + str(args.max_samples) + ' samples from ' + args.train):
        if bidirectional:
            train_dataset = ProteinBindingData(args.train, tokenizer, max_dim=cj['n_ctx'], max_samples=args.max_samples)
        else:
            train_dataset = MaskedProteinData(args.train, tokenizer, max_dim=cj['n_ctx'], max_samples=args.max_samples)
        train_dataloader = make_dataloader(train_dataset)

    print('train samples found:', len(train_dataset))

    # configure training

    num_epochs = args.epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
            name='linear', optimizer=optimizer, num_warmup_steps=5000, num_training_steps=num_training_steps
            )
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.GradScaler(device.type)    

    # train

    model.train()

    step_count = 0
    save_every = args.save_every
    print_every = 1000
    for epoch in range(num_epochs):
        with print_time('\nepoch ' + str(epoch)):
            total_loss = 0
            batches = 0
            for data_batch in train_dataloader:
                # resume from step
                if step_count < start_step:
                    step_count += 1
                    lr_scheduler.step()
                    continue

                if bidirectional:
                    seqs, attns, offsets, targets = data_batch
                    # put everything on the GPU
                    seqs = seqs.to(device)
                    attns = attns.to(device)
                    offsets = offsets.to(device) # TODO: remove if remains unused
                    targets = targets.to(device)
                else:
                    seqs_gt, seqs_masked = data_batch
                    targets = seqs_gt.to(device)
                    seqs = seqs_masked.to(device)
                    attns=None
                    offsets=None
                
                with torch.amp.autocast(device.type):
                    logits = model(seqs,
                                    attention_mask=attns,
                                    pos_offsets=offsets).logits

                    # squish logits + targets, compute loss
                    # TODO: retrieve original lm_head size somehow instead of doing this
                    loss = loss_fn(logits.view(-1, logits.size(-1) // 2), targets.view(-1))
                scaler.scale(loss).backward()

                # unscale then apply gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # print + update loss; if you want the full loss curve over the epoch, remove `end='\r'`
                total_loss += loss.item()
                batches += 1
                print('loss: {:.5f}'.format(total_loss / batches), end='\r')

                if step_count % print_every == 0:
                    print('step {} loss: {:.5f} (this step {:.5f})'.format(step_count, total_loss / batches, loss.item()))

                # save every N steps
                if step_count != start_step and step_count % save_every == 0:
                    save_path = os.path.join(args.save, 'train-' + args.model_type + '-step' + str(step_count) + '.pt')
                    torch.save({
                        'step': step_count,
                        'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict()
                    }, save_path)
                    print('saved to', save_path)
                step_count += 1
            if batches > 0: print('loss: {:.5f}'.format(total_loss / batches))
            
    # save final weights
    save_path = os.path.join(args.save, 'model.pt')
    torch.save(model, save_path)
    print('saved to', save_path, end='\n\n')


if __name__ == '__main__':
    main()
    print('done.')
