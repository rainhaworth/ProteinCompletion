import torch
from torch.utils.data import Dataset
import csv
import numpy as np
from Bio import SeqIO

from .mask import idx_to_mask_start, rand_mask_start

# FASTA reader
# on this branch, assume we are always receiving UniRef data
def fasta_gen(file, start_seq_idx=0):
    idx = 0
    with open(file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq = str(record.seq)
            desc = str(record.description)
            # reject short seqs, low quality seqs
            if len(seq) < 18: continue
            if 'LOW QUALITY PROTEIN' in desc: continue
            # reject non-representatives; false negative if spaces permitted in ID
            uniq_id = desc.split(' ', 1)[0][9:]
            rep_id = desc.rsplit(' ', 1)[1][6:]
            if uniq_id != rep_id: continue
            # if this is a valid sequence, iterate until we reach start_seq_idx
            idx += 1
            if idx <= start_seq_idx: continue
            # output
            yield seq, None

# TSV reader (for UniProt ID mapper output w/ binding sites)
def tsv_gen(file):
    with open(file) as f:
        reader = csv.reader(f, delimiter='\t')
        col_idxs = None
        col_names = ['Sequence', 'Binding site']
        for row in reader:
            # find columns that contain the data we're interested in
            if col_idxs is None:
                col_idxs = [row.index(col_names[0]), row.index(col_names[1])]
                continue
            # grab sequence and raw binding site data string
            seq = row[col_idxs[0]]
            bind = row[col_idxs[1]]
            # parse binding site data
            bind_split = bind.split(';')
            for sub_bind in bind_split:
                # for now, just make pairs for each BINDING instance
                if sub_bind[:7] == 'BINDING':
                    bind_range = sub_bind.split()[-1].split('..')
                    # enforce valid binding site
                    try:
                        # enforce correct number of elements
                        assert 1 <= len(bind_range) <= 2
                        # enforce sequence bounds
                        assert min(0 < int(x) < len(seq) for x in bind_range)
                        # enforce valid range
                        if len(bind_range) == 2:
                            assert int(bind_range[0]) <= int(bind_range[1])
                    except:
                        continue
                    # get single position index or range of position indices
                    if len(bind_range) == 1:
                        # single -> tensor
                        bind_idx = torch.tensor([int(bind_range[0])])
                    else:
                        # range
                        bind_idx = range(int(bind_range[0]), int(bind_range[1])+1)
                        bind_idx = torch.tensor(bind_idx)
                    yield seq, bind_idx

# select generator from file extension
def make_gen_from_ext(file, start=0):
    ext = file.split('.')[-1]
    if ext in ['fasta', 'fa']:
        return fasta_gen(file, start)
    elif ext == 'tsv':
        return tsv_gen(file)
    else:
        raise ValueError('Invalid file extension ' + ext + '; expected fasta or tsv')

# binding site dropout for tensor idxs
def apply_dropout(idxs, p_drop=0.2):
    if len(idxs) <= 1:
        return idxs
    elems_to_drop = np.random.binomial(len(idxs), p_drop)
    elems_to_keep = max(len(idxs) - elems_to_drop, 1)
    idxs_new = idxs[torch.randperm(len(idxs))]
    return torch.sort(idxs_new[:elems_to_keep]).values

class ProteinBindingData(Dataset):
    def __init__(self, file, tokenizer, max_dim=512, max_samples=1000, p_drop=0.2, start_seq_idx=0):
        # start_seq_idx is indexed by order of emission from the generator; usually set to init_step * bsz
        self.max_dim = max_dim
        self.p_drop = p_drop
        # load all data into working memory
        self.seqs = []
        self.idxs = []

        # masking stuff
        self.beta = torch.distributions.beta.Beta(torch.tensor([3.0]), torch.tensor([9.0]))
        self.uniform = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

        # get generator
        gen = make_gen_from_ext(file, start_seq_idx)

        # fetch all sequences and binding sites if available
        sample_count = 0
        for seq, idx in gen:
            # append EOS + BOS markers
            seq = '1' + seq + '2'
            # for long sequences, take random subsequence; best to do this now to avoid unnecessary processing
            if len(seq) > max_dim:
                min_idx = 0
                max_idx = len(seq) - self.max_dim
                offset = np.random.randint(min_idx, max_idx)
                # update seq
                seq = seq[offset : offset + self.max_dim]
                # TODO: handle idxs; for uniref we don't use them
            # tokenize
            seq = tokenizer.encode(seq).ids
            # convert to proper BOS/EOS tokens; see tokenizer.json
            if seq[0] == 3: seq[0] = 1
            if seq[-1] == 4: seq[-1] = 2
            # store
            self.seqs.append(torch.tensor(seq))
            self.idxs.append(idx)

            # have we hit max_samples?
            sample_count += 1
            if sample_count >= max_samples:
                break

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        idxs = self.idxs[idx]

        # generate random path -> mask + targets
        # for now, pad everything to max_dim
        if idxs is not None:
            # apply dropout
            idxs_drop = apply_dropout(idxs, self.p_drop)
            # generate
            attn, targets = idx_to_mask_start(idxs_drop, len(seq), self.max_dim)
        else:
            # compute number of positions allocated to binding site
            choice = self.uniform.sample()
            if choice > 0.8:
                frac = self.uniform.sample()
            else:
                frac = self.beta.sample()
            n_mask = (len(seq) * frac).int()

            # ensure we have at least 1 known position and at least 1 unknown position
            n_mask = max(min(n_mask, len(seq) - 1), 1)

            # make random motif
            rand_idx = torch.randperm(len(seq))[:n_mask]

            attn, targets = idx_to_mask_start(rand_idx, len(seq), self.max_dim)

        # pad sequence
        if len(seq) < self.max_dim:
            seq = torch.cat(( seq, torch.zeros(self.max_dim - len(seq)) )).to(int)

        # convert targets from indices to token ids
        targets = torch.tensor(targets, dtype=int)
        targets = torch.where(targets >= 0, seq[targets], targets)

        # unified training data format: inputs, targets, attns
        return seq, targets, attn

# generation dataset
class ProteinBindingOnlyData(Dataset):
    def __init__(self, file, tokenizer, max_dim=512, max_samples=15, keep_len=False):
        self.max_dim = max_dim
        self.seqs = []
        self.idxs = []

        # set generator type
        gen = make_gen_from_ext(file)
        
        # fetch all
        sample_count = 0
        for seq, idx in gen:
            # tokenize
            seq = tokenizer.encode(seq).ids

            # if keeping full sequence length, enforce bounds now
            if keep_len: seq = seq[:max_dim]

            # generate artificial binding site if necessary
            if idx is None: idx = rand_mask_start(len(seq), self.max_dim, just_binding=True)
            # otherwise, adjust for extra token then randomly drop indices
            else: idx = apply_dropout(idx)
            
            if not keep_len:
                # store smallest possible subsequence
                seq = seq[idx[0] : idx[-1] + 1]
                seq = seq[:max_dim]
                idx -= idx[0]

            # store
            self.seqs.append(torch.tensor(seq))
            self.idxs.append(idx)

            # have we hit max_samples?
            sample_count += 1
            if sample_count >= max_samples:
                break
        # reverse order (temporary)
        self.seqs = self.seqs[::-1]
        self.idxs = self.idxs[::-1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        idxs = self.idxs[idx]

        return seq, idxs


# masked LM
class MaskedProteinData(Dataset):
    def __init__(self, file, tokenizer, max_dim=512, max_samples=1000, start_seq_idx=0):
        # masking stuff
        self.beta = torch.distributions.beta.Beta(torch.tensor([3.0]), torch.tensor([9.0]))
        self.uniform = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

        self.max_dim = max_dim
        # load entire dataset into working memory
        self.seqs = []

        # get generator
        gen = make_gen_from_ext(file, start_seq_idx)

        # copy-pasted from ProteinBindingData
        sample_count = 0
        for seq, _ in gen:
            # append EOS + BOS markers
            seq = '1' + seq + '2'
            # for long sequences, take random subsequence; best to do this now to avoid unnecessary processing
            if len(seq) > max_dim:
                min_idx = 0
                max_idx = len(seq) - self.max_dim
                offset = np.random.randint(min_idx, max_idx)
                # update seq
                seq = seq[offset : offset + self.max_dim]
            # tokenize
            seq = tokenizer.encode(seq).ids
            # convert to proper BOS/EOS tokens; see tokenizer.json
            if seq[0] == 3: seq[0] = 1
            if seq[-1] == 4: seq[-1] = 2
            # store
            self.seqs.append(torch.tensor(seq))

            # have we hit max_samples?
            sample_count += 1
            if sample_count >= max_samples:
                break

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        # if sequence is bigger than max_dim, take random subsequence
        offset = 0
        if len(seq) > self.max_dim:
            min_idx = 0
            max_idx = len(seq) - self.max_dim
            # compute offset
            offset = np.random.randint(min_idx, max_idx)
            # update seq
            seq = seq[offset : offset + self.max_dim]
        
        # make masked sequence
        seq_mask = seq.clone()

        # compute number of positions to mask for weird ESM masking scheme
        choice = self.uniform.sample()
        if choice > 0.8:
            frac = self.uniform.sample()
        else:
            frac = self.beta.sample()
        n_mask = (len(seq) * frac).int()

        # get indices to mask
        mask_idx = torch.randperm(len(seq))[:n_mask]

        # apply; use 3 = <mask> because it's unused
        seq_mask[mask_idx] = 3

        # make target sequence that ignores all non-masked positions
        seq_tgt = torch.zeros(self.max_dim, dtype=int) - 100
        seq_tgt[mask_idx] = seq[mask_idx]

        # pad masked sequence
        if len(seq_mask) < self.max_dim:
            seq_mask = torch.cat(( seq_mask, torch.zeros(self.max_dim - len(seq_mask)) )).to(int)

        # unified training data format: inputs, targets, attns
        return seq_mask, seq_tgt, torch.tensor([])