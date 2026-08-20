import time
import os
import numpy as np
import torch
from tokenizers import Tokenizer
import random
import json

from .config import BaseConfig

class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')

def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def set_seed(seed, deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def create_tokenizer_custom(file: str) -> Tokenizer:
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())
    
# weight initializer
def init_weights(module, config):
    if isinstance(module, (torch.nn.Linear,)):
        # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

# model loading function, ensuring compatibility with old checkpoints
def load_model_compat(model_class, config_file, device, states=None):
    with open(config_file, 'r') as f:
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

    model = model_class(config)
    
    if states is not None:
        # if we have no lm_head in our model, don't try to load
        drop_lm_head = False
        if 'lm_head.weight' not in model.state_dict().keys():
            drop_lm_head = True

        # drop any unused keys
        state_dict = states['model_state']
        keys_to_del = []
        for key in state_dict.keys():
            if 'attn.bias' in key or 'attn.masked_bias' in key or (drop_lm_head and key[:7] == 'lm_head'):
                keys_to_del.append(key)
        for key in keys_to_del:
            del state_dict[key]

        model.load_state_dict(state_dict)
    else:
        print('initializing weights')
        model.apply(lambda x: init_weights(x, config))
    
    model.to(device)
    
    return model

# transformer LR scheduler, from https://huggingface.co/transformers/v4.4.2/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
# we don't use anything else from transformers so this avoids the import entirely
def get_scheduler(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# load all other training state data from config
def load_train_config(model, warmup_steps, train_steps, states=None):
    # initialize
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_scheduler(optimizer, warmup_steps, train_steps)

    # load all provided states
    if states is not None:
        if 'optim_state' in states.keys(): optimizer.load_state_dict(states['optim_state'])
        if 'scheduler_state' in states.keys(): lr_scheduler.load_state_dict(states['scheduler_state'])
        if 'np_rand_state' in states.keys(): np.random.set_state(states['np_rand_state'])
        # torch CPU and CUDA use different rand states; can only ensure determinism if CUDA is always or never available throughout training
        if 'torch_rand_state' in states.keys(): torch.set_rng_state(states['torch_rand_state'])
        if 'torch_cuda_rand_state' in states.keys() and torch.cuda.is_available(): torch.cuda.set_rng_state(states['torch_cuda_rand_state'])

    return optimizer, lr_scheduler