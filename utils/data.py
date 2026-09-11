import torch
from torch.utils.data import Dataset
import csv
import numpy as np
from Bio import SeqIO

from .mask import idx_to_mask_start, rand_mask_start, idx_to_mask_targets_hanoi, diag_block_mask

PAD_ID = 0
SEP_ID = 3

# FASTA reader
# on this branch, assume we are always receiving UniRef data
def fasta_gen(file, start_seq_idx=0):
    idx = 0
    with open(file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            # if this is a valid sequence, iterate until we reach start_seq_idx
            idx += 1
            if idx <= start_seq_idx: continue
            # output
            yield str(record.seq), None

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
                sub_bind = sub_bind.strip()
                # for now, just make pairs for each BINDING instance
                if sub_bind.startswith('BINDING'):
                    bind_range = sub_bind.split()[-1].split('..')
                    # enforce valid binding site
                    try:
                        if not 1 <= len(bind_range) <= 2:
                            raise ValueError
                        coordinates = [int(value) for value in bind_range]
                        if not all(1 <= value <= len(seq) for value in coordinates):
                            raise ValueError
                        if len(coordinates) == 2 and coordinates[0] > coordinates[1]:
                            raise ValueError
                    except (TypeError, ValueError):
                        continue

                    # UniProt feature coordinates are one-based and inclusive.
                    start = coordinates[0] - 1
                    end = coordinates[-1] - 1
                    bind_idx = torch.arange(start, end + 1)
                    yield seq, bind_idx

# select generator from file extension
def make_gen_from_ext(file, start=0):
    ext = str(file).rsplit('.', 1)[-1].lower()
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

# training dataset
class PackedUnirefData(Dataset):
    def __init__(self, file, tokenizer=None, max_dim=512, max_samples=None, p_drop=None, start_seq_idx=None, model_type='atp'):
        # all none arguments are unused, only present to avoid breaking anything
        self.max_dim = max_dim
        self.data_path = file
        self.model_type = model_type
        self.mask_id = None
        if self.model_type == 'esm':
            if tokenizer is None:
                raise ValueError('ESM training requires the tokenizer used to pack the data')
            self.mask_id = tokenizer.token_to_id('<mask>')
            if self.mask_id is None:
                raise ValueError('Tokenizer does not define a <mask> token')
            if self.mask_id in (PAD_ID, SEP_ID):
                raise ValueError('The ESM mask token must be distinct from padding and separators')
        # masking stuff
        self.beta = torch.distributions.beta.Beta(torch.tensor([3.0]), torch.tensor([9.0]))
        self.uniform = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

    def __len__(self):
        return np.memmap(self.data_path, mode='r').size // self.max_dim

    def __getitem__(self, idx):
        # regenerating constantly avoids memory leaks with low overhead, see https://github.com/karpathy/nanoGPT/blob/master/train.py
        data = np.memmap(self.data_path, mode='r')

        # get data
        i = idx * self.max_dim
        seq = torch.from_numpy(data[i:i+self.max_dim].astype(np.int64))

        # extract separators, add extra separator at end or start of padding (easier final block handling)
        last_sep = torch.cat([torch.nonzero(seq == PAD_ID).view(-1), torch.tensor([self.max_dim])])[:1]
        sep_idxs = torch.cat([torch.nonzero(seq == SEP_ID).squeeze(-1), last_sep])

        # pick mask size
        choice = self.uniform.sample()
        if choice > 0.8:
            frac = self.uniform.sample()
        else:
            frac = self.beta.sample()

        # generate mask_idxs with min + max per seq (long, hard to vectorize)

        # assign 1-indexed ids to each seq, 0 to pad/sep
        not_res = (seq == SEP_ID) | (seq == PAD_ID)
        seq_ids = torch.cumsum((seq == SEP_ID), 0) + 1
        seq_ids[not_res] = 0

        # find length of each seq, compute mask size, override pad/sep mask
        all_lens = torch.bincount(seq_ids)
        padsep_len = all_lens[0]
        seq_lens = all_lens[1:]
        mask_szs = (seq_lens * frac).int().clamp(torch.ones_like(seq_lens),seq_lens-1)

        # batch randperm w/ noise overlay; filter out sep/ped
        noise = torch.rand(self.max_dim)
        seq_ids[not_res] = 100
        grouped_noise = noise + (seq_ids * 2.0)
        sorted_idx = torch.argsort(grouped_noise)[:self.max_dim-padsep_len]

        # find seq start positions within sorted_idx, asign 0-indexed seq_ids to each sorted_idx
        start_idx = torch.zeros_like(seq_lens)
        start_idx[1:] = torch.cumsum(seq_lens[:-1], 0)
        seq_ids_sorted = seq_ids[sorted_idx]-1

        # batch randperm slice: assign ranks to each seq in sorted_idx then apply mask_szs thresholds
        seq_ranks = torch.arange(len(sorted_idx)) - start_idx[seq_ids_sorted]
        in_mask = seq_ranks < mask_szs[seq_ids_sorted]
        mask_idxs = sorted_idx[in_mask]

        # end mask_idxs generation

        # final mask + target + attns depend on ATP vs ESM
        if self.model_type == 'atp':
            attn, targets = diag_block_mask(mask_idxs, sep_idxs, self.max_dim)

            # convert targets from indices to token ids
            target_positions = targets >= 0
            token_targets = torch.full_like(targets, -100)
            token_targets[target_positions] = seq[targets[target_positions]]
            targets = token_targets

        else:
            # make target sequence that ignores all non-masked positions
            targets = torch.full((self.max_dim,), -100, dtype=int)
            targets[mask_idxs] = seq[mask_idxs].clone()

            # apply mask to seq
            seq[mask_idxs] = self.mask_id
    
            # make block diagonal attention mask; this is not efficient but it works for ATP
            attn = torch.zeros((self.max_dim, self.max_dim), dtype=torch.uint8)
            start_i = torch.tensor(0)
            for sep_i in sep_idxs:
                seqlen = sep_i-start_i
        
                # add to full
                attn[start_i:sep_i, start_i:sep_i] = torch.ones((seqlen, seqlen), dtype=torch.uint8)
        
                start_i = (sep_i+1).item()

        return seq, targets, attn

# generation dataset
class ProteinBindingOnlyData(Dataset):
    def __init__(
        self,
        file,
        tokenizer,
        max_dim=512,
        max_samples=15,
        keep_len=False,
        motif_dropout=0.0,
    ):
        if not 0 <= motif_dropout <= 1:
            raise ValueError('motif_dropout must be between 0 and 1')
        self.max_dim = max_dim
        self.seqs = []
        self.idxs = []

        # set generator type
        gen = make_gen_from_ext(file)
        
        # fetch all
        sample_count = 0
        for seq, idx in gen:
            raw_length = len(seq)
            # tokenize
            seq = tokenizer.encode(seq).ids
            if len(seq) != raw_length:
                raise ValueError(
                    'Generation requires a tokenizer with one token per residue'
                )

            # generate artificial binding site if necessary
            if idx is None:
                if keep_len:
                    seq = seq[:max_dim]
                idx = torch.as_tensor(
                    rand_mask_start(len(seq), self.max_dim, just_binding=True),
                    dtype=torch.long,
                )
            else:
                idx = torch.as_tensor(idx, dtype=torch.long)
                if motif_dropout > 0:
                    idx = apply_dropout(idx, p_drop=motif_dropout)
                idx = torch.unique(idx, sorted=True)
                if idx.numel() == 0 or idx[0] < 0 or idx[-1] >= len(seq):
                    raise ValueError('Binding-site indices are outside the sequence')

                if keep_len and len(seq) > max_dim:
                    motif_span = int(idx[-1] - idx[0] + 1)
                    if motif_span > max_dim:
                        continue
                    earliest_start = max(0, int(idx[-1]) - max_dim + 1)
                    latest_start = min(int(idx[0]), len(seq) - max_dim)
                    crop_start = (earliest_start + latest_start) // 2
                    seq = seq[crop_start:crop_start + max_dim]
                    idx = idx - crop_start

            if not keep_len:
                motif_span = int(idx[-1] - idx[0] + 1)
                if motif_span > max_dim:
                    continue
                # store smallest possible subsequence
                start = int(idx[0])
                end = int(idx[-1]) + 1
                seq = seq[start:end]
                idx = idx - start

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
