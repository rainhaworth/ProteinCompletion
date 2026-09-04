from pathlib import Path

import pytest


pytest.importorskip('pandas')
pytest.importorskip('matplotlib')
pytest.importorskip('seaborn')
pytest.importorskip('plotly')

from StructureEvaluation import gen_combined_figs_driver
from StructureEvaluation import gen_figs_single_driver


def write_structure_results(path: Path, offset=0.0):
    header = (
        'record_id\tgen_pct\tcontiguous\tppl\tse\tlength\tptm\t'
        'mean_gen_plddt\tmean_non_gen_plddt\tidx\tseq\terror\n'
    )
    rows = []
    record_id = 1
    for gen_pct in (20, 40, 60, 80):
        for contiguous in (True, False):
            rows.append(
                f'{record_id}\t{gen_pct}\t{contiguous}\t3.0\t2.0\t'
                f'{100 + record_id}\t{0.4 + offset + record_id * 0.01}\t'
                f'{0.6 + offset + record_id * 0.01}\t0.7\t[0 2]\tACDE\t\n'
            )
            record_id += 1
    path.write_text(header + ''.join(rows), encoding='utf-8')


def write_original_results(path: Path):
    path.write_text(
        'id\tseq\tptm\tmean_plddt\n'
        'one\tACDE\t0.5\t0.7\n'
        'two\tACDEF\t0.6\t0.8\n',
        encoding='utf-8',
    )


def test_plotting_drivers_create_figures_with_optional_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    atp_path = tmp_path / 'atp.tsv'
    esm_path = tmp_path / 'esm.tsv'
    original_path = tmp_path / 'original.tsv'
    write_structure_results(atp_path)
    write_structure_results(esm_path, offset=0.02)
    write_original_results(original_path)

    combined_dir = tmp_path / 'combined'
    single_dir = tmp_path / 'single'
    assert gen_combined_figs_driver.main(
        [
            '--atp', str(atp_path),
            '--esm', str(esm_path),
            '--original', str(original_path),
            '--output-dir', str(combined_dir),
        ]
    ) == 0
    assert gen_figs_single_driver.main(
        [
            '--input', str(atp_path),
            '--original', str(original_path),
            '--output-dir', str(single_dir),
            '--model-label', 'ATP',
        ]
    ) == 0

    assert list(combined_dir.glob('*.png'))
    assert list(single_dir.glob('*.png'))
