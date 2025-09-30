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
def load_compat(model_class, config_file, device, checkpoint='', training=False):
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
    
    if checkpoint != '' and os.path.exists(checkpoint):
        print('loading from checkpoint', checkpoint)
        states = torch.load(checkpoint, map_location='cpu')
        start_step = states['step']

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
        start_step = 0
        states = None
    
    model.to(device)

    if training:
        # optimizer ends up on cpu if we don't declare after model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        if states is not None: optimizer.load_state_dict(states['optim_state'])

        return model, optimizer, start_step
    
    return model