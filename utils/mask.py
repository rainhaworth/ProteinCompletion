# methods to generate flexible causal masks from sequences
import numpy as np

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
