# base model components from https://github.com/salesforce/progen/blob/main/progen2/models/progen/modeling_progen.py in pure pytorch
# (now heavily modified)
import torch

import torch
import torch.nn as nn

# activation function

class SwiGLU(torch.nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2

# RoPE + attention

def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (seq_len, head_dim/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos.to(dtype), sin.to(dtype)        # (seq_len, head_dim)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    # x: (batch, heads, seq_len, head_dim); cos/sin: (1, 1, seq_len, head_dim)
    return x * cos + rotate_half(x) * sin

class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.pdrop = config.attn_pdrop
        assert self.hidden_size % self.num_heads == 0, f"embed_dim ({self.embed_dim}) must be divisible by num_attention_heads ({self.num_heads})."
        self.head_dim = self.hidden_size // self.num_heads
        
        self.qkv_proj = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self,
            hidden_states,
            attention_mask=None,
            use_cache=False,
            output_attentions=False,):
        # use_cache and output_attentions not currently implemented; torch scaled_dot_product_attention doesn't support mask output
        x = hidden_states
        B, L, D = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(L, self.head_dim, device=x.device, dtype=x.dtype)
        cos, sin = cos[None, None], sin[None, None]  # (1, 1, L, head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask = attention_mask,
            dropout_p = 0.1 if self.training else 0.0,
            is_causal = False
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return (self.o_proj(out), None, None) # output, cache, 

# other model blocks

class MLP(torch.nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = torch.nn.Linear(embed_dim, intermediate_size * 2)
        self.fc_out = torch.nn.Linear(intermediate_size, embed_dim)

        self.act = SwiGLU()
        self.dropout = torch.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU: 8/3 * d, rounded to nearest multiple of 256
        inner_dim = config.n_inner if config.n_inner is not None else round((config.n_embd * 8 / 3) / 256) * 256
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        hidden_states = hidden_states + attn_output

        ffn_output = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + ffn_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)

# full BaseModel (excluding LM head)

class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = torch.nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.h = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = torch.nn.LayerNorm(self.embed_dim)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        output_attentions = output_attentions if output_attentions is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # handle input_ids or input_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.wte(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # create broadcastable + functional attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, :, :]
            #attention_mask = attention_mask.to(dtype=torch.float16)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = None # use_cache not yet implemented
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for block in self.h:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, presents, all_hidden_states, all_self_attentions