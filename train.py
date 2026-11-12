# training script modified from https://github.com/salesforce/progen/blob/main/progen2/sample.py
import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.data import PackedUnirefData, fixed_perm_sampler
from utils.utils import print_time, set_seed, set_env, create_tokenizer_custom, load_model_compat, load_train_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-medlarge.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data', type=str, default='./data/uniprot_sprot.fasta')
    parser.add_argument('--tokenizer', type=str, default='./tokenizer-uniref.json')
    parser.add_argument('--save', type=str, default='./weights')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--total-steps', type=int, default=250000) # specify total training step count for LR scheduling
    parser.add_argument('--warmup-steps', type=int, default=5000)
    parser.add_argument('--save-every', type=int, default=20000)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--model_type', choices=['atp', 'esm'], default='atp')
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
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
            perm_seed = states['perm_seed']
    else:
        states = None
        perm_seed = np.random.randint(np.iinfo(int).max)
        init_step = 0

    # load model, parameters

    with print_time('loading model'):
        model = load_model_compat(model_class, args.config, device, states)

    # load tokenizer, dataset
    
    tokenizer = None
    if args.model_type == 'esm':
        with print_time('loading tokenizer from ' + args.tokenizer):
            tokenizer = create_tokenizer_custom(args.tokenizer)

    def make_dataset_dataloader(data, seed, step):
        dataset = PackedUnirefData(data, tokenizer=tokenizer, max_dim=model.config.n_ctx, model_type=args.model_type, flip_frac=args.flip)
        sampler = fixed_perm_sampler(len(dataset), seed)
        start = step * args.bsz
        if start >= len(dataset): 
            epoch_steps = np.ceil(len(dataset) / args.bsz)
            epochs = step // full_pass_steps
            start = (step - epochs * full_pass_steps) * args.bsz
        sampler = sampler[start:]
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, pin_memory=True, batch_size=args.bsz, sampler=sampler)
        return dataset, dataloader

    with print_time('loading samples from ' + args.data):
        train_dataset, train_dataloader = make_dataset_dataloader(args.data, perm_seed, init_step)

    print('train samples found:', len(train_dataset))

    # configure training

    optimizer, lr_scheduler = load_train_config(model, args.warmup_steps, args.total_steps, states)

    loss_fn = torch.nn.CrossEntropyLoss()

    save_pre = 'train-' + args.model_type
    if args.flip: save_pre += '-flip'

    writer = SummaryWriter()

    model.compile()
    model.train()

    # infinite train loop

    print('training')

    step_count = init_step + 1
    save_every = args.save_every
    print_every = 1000
    flush_every = 50
    while True:
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
            
            # write loss to tensorboard every step, print periodically
            writer.add_scalar('loss', loss.item(), step_count)
            writer.add_scalar('step time', time.time() - t0, step_count)

            if step_count % flush_every == 0: writer.flush()

            if step_count % print_every == 0: print('step {} loss: {:.5f}'.format(step_count, loss.item()))

            # save every N steps
            if step_count % save_every == 0:
                save_path = os.path.join(args.save, save_pre + '-step' + str(step_count) + '.pt')
                torch.save({
                    'step': step_count,
                    'perm_seed': perm_seed,
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

        # if we finish an epoch, reset the dataloader
        perm_seed = np.random.randint(np.iinfo(int).max)
        train_dataset, train_dataloader = make_dataset_dataloader(args.data, perm_seed, 0)


if __name__ == '__main__':
    main()
    print('done.')
