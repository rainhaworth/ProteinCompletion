# methods to generate flexible causal masks from sequences
import numpy as np
import torch
from time import time

# helper function: convert list of indices to list of contiguous segments
# represented as list of tuples (start, end)
def idx_to_segments(idx):
    segments = []
    seg_start = idx[0]
    prev = seg_start
    # iterate over elements past first
    for i in idx[1:]:
        if i - prev == 1:
            # expand segment
            prev = i
        else:
            # complete segment
            segments.append((seg_start, prev))
            # start new segment
            seg_start = i
            prev = i
    # add last segment
    segments.append((seg_start, idx[-1]))

    return segments

# generate path randomly choosing from valid steps
def idx_to_path_targets_valid(idx, seqlen, dim=512):
    path = []
    targets = np.full((dim, 2), -100, dtype=int)
    idx = sorted(idx)

    assert idx[0] >= 0
    assert idx[-1] < seqlen

    # find initial binding site segments
    segments = idx_to_segments(idx)

    # iterate until we only have one segment covering the whole sequence
    while segments[0][0] != 0 or segments[0][1] != seqlen-1:
        # pick a random segment + direction
        i = np.random.randint(0, len(segments))
        direction = np.random.randint(2)
        pos = segments[i][direction]
        # enforce bounds
        if (pos <= 0 and direction == 0) or (pos >= seqlen-1 and direction == 1):
            direction = 1 - direction
            pos = segments[i][direction]
        off = (direction*2) - 1

        # update targets + path
        targets[pos, direction] = pos + off # off=-1 -> targets[pos, 0]; off=1 -> targets[pos, 1]
        # if we already have a target at this location, retroactively predict both targets simultaneously (necessary for masking)
        if targets[pos, 1 - direction] == -100: path.append(pos) # off=-1 -> targets[pos,1]; off=1 -> targets[pos,0]

        # update segments
        if off == -1:
            # merge segments if necessary; no new targets because both adjacent amino acids are known
            if i > 0 and segments[i-1][1] - off == pos + off:
                segments[i] = (segments[i-1][0], segments[i][1])
                segments.pop(i-1)
            # otherwise expand segment and update targets
            else:
                segments[i] = (pos + off, segments[i][1])
        else:
            # same as before but opposite direction
            if i < len(segments)-1 and segments[i+1][0] - off == pos + off:
                segments[i] = (segments[i][0], segments[i+1][1])
                segments.pop(i+1)
            else:
                segments[i] = (segments[i][0], pos + off)
    
    return path, targets

# from path, i.e. sequence of indices representing steps, and indices of known monomers, generate mask
def path_to_mask(path, targets, idx, dim=512):
    mask = np.zeros((dim, dim), dtype=np.uint8)

    # for each index in original binding site, unmask the entire column
    mask[:len(targets), idx] = 1

    # fast version, only 2 vectorized updates
    path = np.array(path, dtype=int)
    tp = targets[path,:]
    tp_max = tp.max(1)
    tp_min = tp.min(1)
    # populate upper left triangle
    mask[path[::-1][:,None], tp_max[None,:]] = np.tri(len(path),k=-1,dtype=int)[::-1,:]
    # cleanup: for each step in the path with 2 targets, we missed a column
    ti = np.nonzero(tp_min != -100)[0]
    if len(ti) > 0: mask[:, tp_min[ti]] = mask[:, tp_max[ti]]

    # simple version
    """
    path = np.array(path, dtype=int)
    for i in range(len(path)-1):
        # for all future steps in path, reveal current targets
        to_pop = path[i+1:]
        for t in targets[path[i]]:
            if t == -100: continue
            mask[to_pop, t] = 1"""
    
    # iterative version (use if reimplementing in compiled language)
    """
    for i in range(len(path)-1):
        for j in range(i+1, len(path)):
            for t in targets[path[i]]:
                if t == -100: continue
                mask[path[j], t] = 1
    """
    return mask

# from known indices and sequence length, generate mask and return binding site start position
# new: also generate targets
def idx_to_mask_start(idx, seqlen, dim=512, pathfn=idx_to_path_targets_valid):
    assert 0 < len(idx) <= seqlen
    assert seqlen <= dim
    
    path, targets = pathfn(idx, seqlen, dim)
    mask = path_to_mask(path, targets, idx, dim)

    return mask, targets

# generate random path through sequence of known length
# just_binding arg: skip making the mask, just return the binding site
def rand_mask_start(seqlen, dim=512, exp_sz=5, p_drop=0.2, just_binding=False):
    # generate artificial binding site position
    sz = min(max(1, np.random.poisson(exp_sz)), seqlen-1)
    keep_idx = np.random.random(sz) > p_drop
    if np.sum(keep_idx) == 0:
        keep_idx[0] = True
    start = np.random.randint(0, seqlen-sz)
    idx = np.arange(start, start+sz)[keep_idx]
    
    if just_binding:
        return idx

    # get mask and start position
    return idx_to_mask_start(idx, seqlen, dim)


def idx_to_segments_tensor(idx):
    breaks = idx.diff() != 1
    seg_starts = torch.cat([torch.tensor([0]), breaks.nonzero().squeeze(-1) + 1])
    seg_ends = torch.cat([seg_starts[1:]-1, torch.tensor([idx.size(0)-1])])
    return torch.stack([idx[seg_starts], idx[seg_ends]], 1).to(int)

# leakage-free idea 1: perform all possible generation steps at each time step
# kinda looks like towers of hanoi
# also skip the path since it's deterministic, just generate mask
def idx_to_mask_targets_hanoi(idx, dim=512):
    assert 0 < idx.size(0) <= dim

    targets = torch.full((dim, 2), -100, dtype=int)
    mask = torch.zeros((dim, dim), dtype=torch.uint8)
    mask[:dim, idx] = 1
    
    idx = torch.sort(idx)[0]

    assert idx[0] >= 0
    assert idx[-1] < dim

    # find initial motif segments
    segments = idx_to_segments_tensor(idx)
    visited = idx.clone()

    # iterate until we only have one segment covering the whole sequence
    while visited.size(0) < dim:
        # get all NTP and PTP targets
        segments = idx_to_segments_tensor(idx)
        n_pos_s = segments[:,1]
        p_pos_s = segments[:,0]

        # enforce bounds
        n_pos_s = n_pos_s[n_pos_s < dim-1]
        p_pos_s = p_pos_s[p_pos_s > 0]

        # update targets; this should work even when one is empty
        gen_n = n_pos_s + 1
        gen_p = p_pos_s - 1
        targets[n_pos_s, 1] = gen_n
        targets[p_pos_s, 0] = gen_p

        # update mask; best to work 1 step ahead or we lose non-visited generation targets
        gen = torch.cat([gen_n, gen_p])
        visited = torch.cat([visited, gen])
        new_row = torch.zeros(dim, dtype=torch.uint8)
        new_row[visited] = 1
        mask[gen,:] = new_row

        # update idx
        idx = torch.nonzero(new_row).squeeze().to(idx.dtype)
        

    return mask, targets

# TODO: unit test
def diag_block_mask(mask_idxs, sep_idxs, dim=512, i2m_f=idx_to_mask_targets_hanoi):
    full_mask = torch.zeros((dim, dim), dtype=torch.uint8)
    full_tgts = torch.full((dim, 2), -100, dtype=int)

    start_i = torch.tensor(0)
    for sep_i in sep_idxs:
        mask_idxs_ = mask_idxs[mask_idxs >= start_i]
        mask_idxs_ = mask_idxs_[mask_idxs_ < sep_i]
        if mask_idxs_.size(0) == 0: continue
        sub_mask, sub_tgts = i2m_f(mask_idxs_-start_i, sep_i-start_i)

        # add to full
        sub_tgts[sub_tgts>0] += start_i
        full_mask[start_i:sep_i, start_i:sep_i] = sub_mask
        full_tgts[start_i:sep_i, :] = sub_tgts

        start_i = (sep_i+1).item()

    return full_mask, full_tgts

def profile_path_mask(idx, seqlen, dim=512, pathfn=idx_to_path_targets_valid):
    assert 0 < len(idx) <= seqlen
    assert seqlen <= dim
    
    t0 = time()
    path, targets = pathfn(idx, seqlen, dim)
    t1 = time()
    mask = path_to_mask(path, targets, idx, dim)
    t2 = time()

    return t1-t0, t2-t1