# training script modified from https://github.com/salesforce/progen/blob/main/progen2/sample.py
import os
import argparse
import torch
import numpy as np
import time

from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.data import PackedUnirefData
from utils.utils import print_time, set_seed, set_env, create_tokenizer_custom, load_model_compat, load_train_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-medium')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data', type=str, default='./data/uniprot_sprot.fasta')
    parser.add_argument('--save', type=str, default='./weights')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--total-steps', type=int, default=250000) # specify total training step count for LR scheduling
    parser.add_argument('--warmup-steps', type=int, default=5000)
    parser.add_argument('--save-every', type=int, default=20000)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--model_type', choices=['atp', 'esm'], default='atp')
    args = parser.parse_args()

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    configf = f'./{args.config}.json'
    checkpoint = args.ckpt
    if args.model_type == 'atp':
        model_class = BidirectionalCausalLM
        
    else:
        model_class = ESMlikeLM

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False
        
    # load checkpoint if provided
    if checkpoint != '' and os.path.exists(checkpoint):
        with print_time('loading checkpoint data from ' + checkpoint):
            states = torch.load(checkpoint, map_location='cpu', weights_only=False)
            init_step = states['step']
    else:
        states = None
        init_step = 0

    # load model, parameters

    with print_time('loading model'):
        model = load_model_compat(model_class, configf, device, states)

    # load dataset(s)
    
    def make_dataloader(dataset):
        return torch.utils.data.DataLoader(dataset, num_workers=2, pin_memory=True, batch_size=args.bsz, shuffle=True)

    with print_time('loading samples from ' + args.data):
        train_dataset = PackedUnirefData(args.data, max_dim=model.config.n_ctx, model_type=args.model_type)
        train_dataloader = make_dataloader(train_dataset)

    print('train samples found:', len(train_dataset))

    # configure training

    num_epochs = args.epochs
    if args.total_steps != -1: num_training_steps = args.total_steps
    else: num_training_steps = num_epochs * len(train_dataloader)

    optimizer, lr_scheduler = load_train_config(model, args.warmup_steps, num_training_steps, states)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.compile()
    model.train()

    # train

    step_count = init_step + 1
    save_every = args.save_every
    print_every = save_every//2
    for epoch in range(num_epochs):
        with print_time('\nepoch ' + str(epoch)):
            total_loss = 0
            batches = 0
            t0 = time.time()
            for seqs, targets, attns in train_dataloader:
                # put everything on the GPU
                seqs = seqs.to(device)
                targets = targets.to(device)
                if attns.shape[1] == 0:
                    attns = None
                else:
                    attns = attns.to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16): # using ampere gpu; otherwise set to fp16 and use gradient scaling
                    logits = model(seqs, attention_mask=attns)
                    loss = loss_fn(logits.view(-1, model.config.vocab_size), targets.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # print + update loss
                total_loss += loss.item()
                batches += 1
                print('loss: {:.5f}\ttime: {:.4f}s'.format(total_loss / batches, time.time() - t0), end='\r')

                if step_count % print_every == 0:
                    print('step {} loss: {:.5f} (this step {:.5f})'.format(step_count, total_loss / batches, loss.item()))

                # save every N steps
                if step_count % save_every == 0:
                    save_path = os.path.join(args.save, 'train-' + args.model_type + '-step' + str(step_count) + '.pt')
                    torch.save({
                        'step': step_count,
                        'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict(),
                        'scheduler_state': lr_scheduler.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'torch_rand_state': torch.get_rng_state(),
                        'torch_cuda_rand_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    }, save_path)
                    print('saved to', save_path)
                step_count += 1
                t0 = time.time()
            if batches > 0: print('loss: {:.5f}'.format(total_loss / batches))
            
    # save final weights
    save_path = os.path.join(args.save, 'model.pt')
    torch.save(model, save_path)
    print('saved to', save_path, end='\n\n')


if __name__ == '__main__':
    main()
    print('done.')
