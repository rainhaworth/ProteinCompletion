import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from utils.data import ProteinBindingOnlyData, tsv_gen


def make_tokenizer():
    tokens = ['<pad>', '<bos>', '<eos>', '<sep>'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    vocab = {token: index for index, token in enumerate(tokens)}
    vocab['<unk>'] = len(vocab)
    vocab['<mask>'] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, '<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Split('', 'isolated')
    return tokenizer


def test_uniprot_binding_coordinates_are_converted_to_zero_based_indices(tmp_path):
    path = tmp_path / 'binding.tsv'
    path.write_text(
        'Sequence\tBinding site\n'
        'ACDEF\tBINDING 2..4\n'
        'ACDEF\tBINDING 1; BINDING 5\n',
        encoding='utf-8',
    )

    rows = list(tsv_gen(path))

    assert torch.equal(rows[0][1], torch.tensor([1, 2, 3]))
    assert torch.equal(rows[1][1], torch.tensor([0]))
    assert torch.equal(rows[2][1], torch.tensor([4]))


def test_esm_crop_keeps_a_binding_site_near_the_sequence_end(tmp_path):
    path = tmp_path / 'binding.tsv'
    path.write_text(
        'Sequence\tBinding site\n'
        'ACDEFGHIKLMN\tBINDING 11\n',
        encoding='utf-8',
    )
    tokenizer = make_tokenizer()

    dataset = ProteinBindingOnlyData(
        path,
        tokenizer,
        max_dim=5,
        max_samples=1,
        keep_len=True,
    )
    sequence, indices = dataset[0]

    assert len(sequence) == 5
    assert indices.tolist() == [4]
    assert tokenizer.id_to_token(int(sequence[indices[0]])) == 'M'


def test_supplied_binding_motif_is_not_dropped_by_default(tmp_path):
    path = tmp_path / 'binding.tsv'
    path.write_text(
        'Sequence\tBinding site\n'
        'ACDEFGHIK\tBINDING 2..8\n',
        encoding='utf-8',
    )

    dataset = ProteinBindingOnlyData(
        path,
        make_tokenizer(),
        max_dim=9,
        max_samples=1,
        keep_len=True,
    )
    _, indices = dataset[0]

    assert indices.tolist() == [1, 2, 3, 4, 5, 6, 7]
