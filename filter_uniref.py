from Bio import SeqIO
from tqdm import tqdm

f_in = './data/uniref50.fasta'
f_out = './data/uniref50-trimmed.fasta'
n_seq = 60315044

seqs = []
for record in tqdm(SeqIO.parse(f_in, 'fasta'), total=n_seq):
    seq = str(record.seq)
    desc = str(record.description)
    # reject short seqs, low quality seqs
    if len(seq) < 18: continue
    if 'LOW QUALITY PROTEIN' in desc: continue
    # reject non-representatives; false negative if spaces permitted in ID
    uniq_id = desc.split(' ', 1)[0][9:]
    rep_id = desc.rsplit(' ', 1)[1][6:]
    if uniq_id != rep_id: continue
    seqs.append(record)

SeqIO.write(seqs, f_out, 'fasta')