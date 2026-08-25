from pathlib import Path

import pytest

from utils.data import (
    detect_fasta_format,
    make_gen_from_ext,
    require_nonempty_dataset,
)


def write_fasta(path: Path, records):
    path.write_text(
        ''.join(f'>{header}\n{sequence}\n' for header, sequence in records),
        encoding='utf-8',
    )


def test_auto_format_reads_ordinary_fasta_and_honors_start(tmp_path):
    fasta = tmp_path / 'swissprot.fasta'
    records = [
        ('sp|P00001|FIRST Protein one', 'ACDE'),
        ('sp|P00002|SECOND Protein two', 'FGHI'),
    ]
    write_fasta(fasta, records)

    assert detect_fasta_format(fasta) == 'fasta'
    assert list(make_gen_from_ext(fasta, start=1)) == [('FGHI', None)]


def test_auto_format_preserves_current_uniref_filter(tmp_path):
    fasta = tmp_path / 'uniref50.fasta'
    records = [
        ('UniRef50_Q9K794 Example n=1 RepID=Q9K794', 'ACDEFGHIKLMNPQRSTV'),
        ('UniRef50_P12345 Example n=1 RepID=DIFFERENT', 'TVWYACDEFGHIKLMNPQ'),
    ]
    write_fasta(fasta, records)

    assert detect_fasta_format(fasta) == 'uniref'
    assert list(make_gen_from_ext(fasta)) == [('ACDEFGHIKLMNPQRSTV', None)]


def test_explicit_fasta_format_bypasses_uniref_header_filter(tmp_path):
    fasta = tmp_path / 'records.fasta'
    write_fasta(
        fasta,
        [('UniRef50_P12345 Example n=1 RepID=DIFFERENT', 'ACDE')],
    )

    assert list(make_gen_from_ext(fasta, data_format='fasta')) == [('ACDE', None)]


def test_empty_fasta_and_empty_dataset_fail_clearly(tmp_path):
    empty_fasta = tmp_path / 'empty.fasta'
    empty_fasta.write_text('', encoding='utf-8')

    with pytest.raises(ValueError, match='No FASTA records'):
        make_gen_from_ext(empty_fasta)

    with pytest.raises(ValueError, match='No training sequences'):
        require_nonempty_dataset([], empty_fasta)
