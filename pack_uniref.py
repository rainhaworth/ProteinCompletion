# best fit decreasing bin packing
from Bio import SeqIO
from tqdm import tqdm
import random
import bisect
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

f_in = './data/uniref50-trimmed.fasta'
f_out = './data/uniref50-packed.bin'
max_cap = 1024

random.seed(42)


print('initializing tokenizer')


# 4 special tokens
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
TOK_START = 4
tokens = ['<pad>', '<bos>', '<eos>', '<sep>'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
vocab = {token: i for i, token in enumerate(tokens)}
vocab['<unk>'] = len(vocab)
tokenizer = Tokenizer(models.WordLevel(vocab, '<unk>'))
tokenizer.pre_tokenizer = pre_tokenizers.Split('', 'isolated') # character-level
tokenizer.save('tokenizer-uniref.json')


print('loading sequences')


seqs = []
i = 0
for record in tqdm(SeqIO.parse(f_in, 'fasta')):
    # manually tokenize + add EOS, BOS tokens (this is faster and somehow less clunky than the tokenizer)
    seq = chr(BOS_ID + ord('A') - TOK_START) + str(record.seq) + chr(EOS_ID + ord('A') - TOK_START)
    if len(seq) > max_cap:
        start = random.randint(0, len(seq)-max_cap)
        seq = seq[start:start+max_cap]
    ids = [ord(x) - ord('A') + TOK_START for x in seq]
    seqs.append(ids)

# sort by length, descending
seqs = sorted(seqs, key=len, reverse=True)
min_len = len(seqs[-1])
assert len(seqs[0]) <= max_cap


print('packing')


closed_bins = [] # list of seqs
open_bins = [] # list of (seq, capacity)
for seq in tqdm(seqs):
    if len(seq) == max_cap:
        closed_bins.append(seq)
    elif len(open_bins) == 0 or len(seq) > (max_cap//2) + 1: # we know none of these seqs will be packable
        open_bins.append((seq, max_cap-len(seq)))
    else:
        # find best fit
        # figuring out binary search version
        best_i = bisect.bisect_left(open_bins, len(seq)+1, key=lambda x: x[1])

        # add to bin or make new
        if best_i != len(open_bins):
            # add SEP token
            new_seq = open_bins[best_i][0] + [SEP_ID] + seq
            new_gap = max_cap - len(new_seq)

            open_bins.pop(best_i)

            # prune if full
            if new_gap < min_len:
                closed_bins.append(new_seq)
            # otherwise find insertion point
            else:
                in_i = bisect.bisect_left(open_bins, new_gap, key=lambda x: x[1])
                open_bins.insert(in_i, (new_seq, new_gap))
        else:
            # this is guaranteed to be the smallest seq we have seen so far
            open_bins.append((seq, max_cap-len(seq)))

# add any remaining open bins to closed bins
for b in open_bins:
    closed_bins.append(b[0])


print('writing')


# memmap so we don't have to read in everything
arr = np.memmap(f_out, mode='w+', shape=(len(closed_bins),max_cap))
for i, seq in tqdm(enumerate(closed_bins)):
    # pad
    arr[i,:] = seq + [PAD_ID] * (max_cap - len(seq))
arr.flush()


print('done')