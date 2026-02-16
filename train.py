# training script modified from https://github.com/salesforce/progen/blob/main/progen2/sample.py
import os
import argparse
import torch

from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.data import ProteinBindingData, MaskedProteinData
from utils.utils import print_time, set_seed, set_env, create_tokenizer_custom, load_compat, get_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config-medium')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data', type=str, default='./data/uniprot_sprot.fasta')
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
    if args.model_type == 'bidirectional':
        model_class = BidirectionalCausalLM
        data_class = ProteinBindingData
    else:
        model_class = ESMlikeLM
        data_class = MaskedProteinData

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # load model, parameters, tokenizer

    with print_time('loading parameters'):
        model, optimizer, start_step = load_compat(model_class, configf, device, ckpt, training=True)

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    # load dataset(s)
    
    # helper function; keep it small and simple for now
    def make_dataloader(dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=args.bsz)

    with print_time('loading up to ' + str(args.max_samples) + ' samples from ' + args.data):
        train_dataset = data_class(args.data, tokenizer, max_dim=model.config.n_ctx, max_samples=args.max_samples)
        train_dataloader = make_dataloader(train_dataset)

    print('train samples found:', len(train_dataset))

    # configure training

    num_epochs = args.epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(optimizer, 5000, num_training_steps)

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
            for seqs, targets, attns in train_dataloader:
                # resume from step
                if step_count < start_step:
                    step_count += 1
                    lr_scheduler.step()
                    continue

                # put everything on the GPU
                seqs = seqs.to(device)
                targets = targets.to(device)
                if attns is not None:
                    attns = attns.to(device)
                
                with torch.amp.autocast(device.type):
                    logits = model(seqs, attention_mask=attns)

                    # squish logits + targets, compute loss
                    loss = loss_fn(logits.view(-1, model.config.vocab_size), targets.view(-1))
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
