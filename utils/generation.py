# generation utils
import torch
from utils.mask import idx_to_segments

# TODO: specify these in config, streamline generation functions
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# from sequence + list of valid indices, generate mask for inference
def make_inference_mask(seqlen, idx, device, dim=512):
    # reduce to subsequence if necessary
    assert idx[-1] < seqlen
    assert idx[0] >= 0

    # make mask
    sz = min(seqlen, dim) # TODO: handle reaching max dim better
    mask = torch.zeros((sz,sz)).to(device)
    mask[:,idx] = 1 # unmask entire columns

    # add batch dim
    return mask[None,:,:]

# greedy sampling: find best logit and return corresponding position + token
def greedy_sample(logits):
    vals, toks = torch.max(logits, dim=-1)
    best_i = torch.argmax(vals)
    return best_i, toks[best_i]

# nucleus sampling: choose position + token with probability given by logit distribution
def nucleus_sample(logits, p=0.95):
    # find largest cutoff where we retain at least p of the probability mass
    logits_rev = logits.flatten().sort(descending=True)[0]
    cum_probs = logits_rev.cumsum(0)
    min_keep_idx = torch.sum(cum_probs < p)

    # rescale logits
    min_keep_val = logits_rev[min_keep_idx]
    p_prime = cum_probs[min_keep_idx]
    logits_rescaled = torch.where(logits >= min_keep_val, logits / p_prime, 0)

    # sample; need to flatten then convert back to dim 0, dim 1 indices
    idx_flat = torch.multinomial(logits_rescaled.flatten(), 1)[0]
    return idx_flat // logits.shape[1], idx_flat % logits.shape[1]

# causal bidirectional generation
def gen_step_bidirectional(model, seq, idxs, device, invalid_ids=[], rp=1.2, rw=4, sample_fn=nucleus_sample, return_logits=False, predict_terminals=True):
    # get segments; use copy of idxs so we don't have weird memory issues
    segments = idx_to_segments(idxs.detach().clone())

    # get PTP/NTP indices
    p_idxs = [seg[0] for seg in segments if seq[:,seg[0]] not in [BOS_ID, BOS_ID+2]]
    n_idxs = [seg[1] for seg in segments if seq[:,seg[1]] not in [EOS_ID, EOS_ID+2]]

    # if not predicting terminals, assume fixed window like ESM
    if not predict_terminals:
        if len(p_idxs) > 0 and p_idxs[0] == 0: p_idxs.pop(0)
        if len(n_idxs) > 0 and n_idxs[-1] == seq.size(1) - 1: n_idxs.pop(-1)

    # stop inference if we have no valid steps
    if len(n_idxs) == 0 and len(p_idxs) == 0: return None, None

    # make mask, call model, squeeze batch dim to make life easier
    mask = make_inference_mask(seq.size(1), idxs, device, seq.size(1))
    logits = model(seq, attention_mask=mask)
    logits = torch.squeeze(logits, 0)

    # get PTP/NTP logits
    half_sz = logits.size(-1) // 2
    p_logits = logits[p_idxs,:half_sz]
    n_logits = logits[n_idxs,half_sz:]

    # concat logits
    logits = torch.concat([p_logits, n_logits])  

    # apply repetition penalties; can probably do this faster but with window=4 it's fine
    p_penalties = [[seq[:,i] for i in range(p_i, p_i+rw) if i in idxs] for p_i in p_idxs]
    n_penalties = [[seq[:,i] for i in range(n_i-rw+1, n_i+1) if i in idxs] for n_i in n_idxs]
    penalties = torch.ones_like(logits)
    for i, pens in enumerate(p_penalties + n_penalties):
        for p in pens:
            penalties[i, p] = rp
    logits = logits / penalties

    # make + apply logit mask so we don't generate invalid tokens
    drop_val = -1e9
    mask = torch.zeros_like(logits)
    mask[:,invalid_ids] = drop_val
    if predict_terminals:
        # if we can predict BOS, allow
        if len(p_idxs) > 0 and p_idxs[0] == 0:
            mask[0, [BOS_ID, BOS_ID+2]] = 0
        # same for EOS
        if len(n_idxs) > 0 and n_idxs[-1] == seq.size(1) - 1:
            mask[-1, [EOS_ID, EOS_ID+2]] = 0
    logits += mask

    # compute (numerically stable) softmax over all logits representing viable next steps
    exp_logits = torch.exp(logits - torch.max(logits))
    sum_exp_logits = torch.sum(exp_logits)
    logits = exp_logits / sum_exp_logits

    if return_logits: return logits, None

    # sample next step (index, token) from logits
    new_i, new_token = sample_fn(logits)
    PTP = new_i < len(p_idxs)

    # get new token position from index, offset according to PTP vs NTP
    if PTP: new_pos = p_idxs[new_i] - 1
    else:   new_pos = n_idxs[new_i - len(p_idxs)] + 1

    return new_token, new_pos

# esmlike generation
def gen_step_esmlike(model, seq, idxs, device, invalid_ids=[], rp=1.2, rw=4, sample_fn=nucleus_sample, return_logits=False, predict_terminals=None):
    # unimplemented: repetition penalties, predict_terminals

    # stop generation if we run out of positions to predict
    if len(idxs) == len(seq): return None, None

    # get logits (L,V)
    mask = make_inference_mask(seq.size(1), idxs, device, seq.size(1))
    logits = model(seq, attention_mask=mask)
    logits = torch.squeeze(logits, 0)

    # mask invalid
    drop_val = -1e9
    mask = torch.zeros_like(logits)
    mask[:,invalid_ids] = drop_val
    logits += mask

    # softmax
    logits_softmax = torch.nn.functional.softmax(logits, -1)

    if return_logits: return logits_softmax, None

    # get entropy at each position (L), pick inference position with lowest entropy
    entropy = torch.distributions.Categorical(logits=logits.log_softmax(-1)).entropy()
    entropy[torch.squeeze(idxs)] = 1e9
    new_pos = torch.argmin(entropy)

    # get new token
    logits = logits_softmax[new_pos,:]
    _, new_token = sample_fn(logits[None,:])

    return new_token, new_pos