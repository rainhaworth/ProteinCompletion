import csv
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

import eval as eval_script
import generate as generate_script
from StructureEvaluation import gen_combined_figs_driver
from StructureEvaluation import gen_figs_single_driver
from StructureEvaluation import structure_evaluator_driver
from StructureEvaluation.structure_evaluator import StructureEvaluator
from utils.config import BaseConfig
from utils.evaluation import score_atp_logits, score_esm_logits
from utils.generation import update_generation_state
from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM


def tiny_config():
    return BaseConfig(
        vocab_size=32,
        n_positions=8,
        n_ctx=8,
        n_embd=64,
        n_layer=1,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )


def write_tiny_checkpoint(tmp_path):
    config_path = tmp_path / 'tiny.json'
    config_path.write_text(
        json.dumps(
            {
                'vocab_size': 32,
                'n_positions': 8,
                'n_ctx': 8,
                'n_embd': 64,
                'n_layer': 1,
                'n_head': 8,
                'resid_pdrop': 0.0,
                'embd_pdrop': 0.0,
                'attn_pdrop': 0.0,
            }
        ),
        encoding='utf-8',
    )
    model = ESMlikeLM(tiny_config())
    checkpoint_path = tmp_path / 'tiny.pt'
    torch.save({'step': 0, 'model_state': model.state_dict()}, checkpoint_path)
    return config_path, checkpoint_path


def write_uniref_tokenizer(tmp_path):
    tokens = ['<pad>', '<bos>', '<eos>', '<sep>'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    vocab = {token: index for index, token in enumerate(tokens)}
    vocab['<unk>'] = len(vocab)
    vocab['<mask>'] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, '<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Split('', 'isolated')
    tokenizer_path = tmp_path / 'tokenizer-uniref.json'
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


def test_scoring_helpers_align_targets_and_heads():
    sequence = torch.tensor([[4, 5, 6]])
    atp_logits = torch.full((1, 3, 64), -20.0)
    atp_logits[0, 1, 4] = 20
    atp_logits[0, 2, 5] = 20
    atp_logits[0, 0, 32 + 5] = 20
    atp_logits[0, 1, 32 + 6] = 20

    atp_result = score_atp_logits(sequence, atp_logits)
    esm_result = score_esm_logits(
        sequence,
        torch.nn.functional.one_hot(sequence, 32).float() * 20,
        [0, 2],
    )

    assert atp_result['accuracy'] == 1
    assert atp_result['targets'] == 4
    assert esm_result['accuracy'] == 1
    assert esm_result['targets'] == 2


def test_generation_state_handles_prepend_replace_and_append():
    sequence = torch.tensor([[5, 6]])
    known = torch.tensor([0, 1])

    sequence, known = update_generation_state(sequence, known, 4, -1)
    sequence, known = update_generation_state(sequence, known, 7, sequence.size(1))
    sequence, known = update_generation_state(sequence, known, 8, 1)

    assert torch.equal(sequence, torch.tensor([[4, 8, 6, 7]]))
    assert torch.equal(known, torch.arange(4))


def test_generate_and_eval_scripts_run_with_tiny_checkpoint(tmp_path):
    config_path, checkpoint_path = write_tiny_checkpoint(tmp_path)
    tokenizer_path = write_uniref_tokenizer(tmp_path)
    fasta_path = tmp_path / 'proteins.fasta'
    fasta_path.write_text('>example\nACDEFG\n', encoding='utf-8')
    generation_output = tmp_path / 'generated.tsv'
    evaluation_output = tmp_path / 'evaluated.tsv'

    generate_script.main(
        [
            '--weights', str(checkpoint_path),
            '--config', str(config_path),
            '--data', str(fasta_path),
            '--tokenizer', str(tokenizer_path),
            '--device', 'cpu',
            '--model_type', 'esm',
            '--sample', 'greedy',
            '--max-samples', '1',
            '--output', str(generation_output),
        ]
    )
    eval_script.main(
        [
            '--weights', str(checkpoint_path),
            '--config', str(config_path),
            '--data', str(fasta_path),
            '--tokenizer', str(tokenizer_path),
            '--device', 'cpu',
            '--model_type', 'esm',
            '--max-samples', '1',
            '--output', str(evaluation_output),
        ]
    )

    generated = list(csv.DictReader(generation_output.open(), delimiter='\t'))
    evaluated = list(csv.DictReader(evaluation_output.open(), delimiter='\t'))
    assert generated[0]['complete'] == 'True'
    assert generated[0]['sequence']
    assert evaluated[0]['targets'] == '1'


def test_generate_and_eval_scripts_run_for_atp(tmp_path):
    config_path, _ = write_tiny_checkpoint(tmp_path)
    tokenizer_path = write_uniref_tokenizer(tmp_path)
    checkpoint_path = tmp_path / 'tiny-atp.pt'
    torch.save(
        {'step': 0, 'model_state': BidirectionalCausalLM(tiny_config()).state_dict()},
        checkpoint_path,
    )
    fasta_path = tmp_path / 'proteins.fasta'
    fasta_path.write_text('>example\nACDEFG\n', encoding='utf-8')
    generation_output = tmp_path / 'generated-atp.tsv'
    evaluation_output = tmp_path / 'evaluated-atp.tsv'

    generate_script.main(
        [
            '--weights', str(checkpoint_path),
            '--config', str(config_path),
            '--data', str(fasta_path),
            '--tokenizer', str(tokenizer_path),
            '--device', 'cpu',
            '--model_type', 'atp',
            '--sample', 'greedy',
            '--max-steps', '2',
            '--max-samples', '1',
            '--output', str(generation_output),
        ]
    )
    eval_script.main(
        [
            '--weights', str(checkpoint_path),
            '--config', str(config_path),
            '--data', str(fasta_path),
            '--tokenizer', str(tokenizer_path),
            '--device', 'cpu',
            '--model_type', 'atp',
            '--max-samples', '1',
            '--output', str(evaluation_output),
        ]
    )

    generated = next(csv.DictReader(generation_output.open(), delimiter='\t'))
    evaluated = next(csv.DictReader(evaluation_output.open(), delimiter='\t'))
    assert generated['sequence']
    assert int(evaluated['targets']) == 10


class FakeGenerationConfig:
    def __init__(self, track, num_steps):
        self.track = track
        self.num_steps = num_steps


class FakeProtein:
    def __init__(self, sequence):
        self.sequence = sequence

    def to_pdb(self, path):
        Path(path).write_text('PDB', encoding='utf-8')


class FakeClient:
    def generate(self, protein, config):
        assert config.track == 'structure'
        protein.ptm = torch.tensor(0.6)
        protein.plddt = torch.tensor([0.1, 0.2, 0.3, 0.4])
        return protein


def test_structure_evaluator_supports_offline_client(tmp_path):
    evaluator = StructureEvaluator(
        'fake',
        client=FakeClient(),
        protein_class=FakeProtein,
        generation_config_class=FakeGenerationConfig,
    )

    result = evaluator.generate_structure(
        'ACDE', [0, 2], tmp_path / 'protein.pdb'
    )

    assert torch.allclose(torch.tensor(result), torch.tensor([0.6, 0.3, 0.2]))
    assert (tmp_path / 'protein.pdb').is_file()


class FakeEvaluator:
    def generate_structure(self, sequence, non_generated_indices, pdb_out):
        Path(pdb_out).write_text('PDB', encoding='utf-8')
        return 0.5, 0.7, 0.8


def write_completion_tsv(path):
    path.write_text(
        'generated %\tcontiguous\tfull-sequence PPL\tAA composition entropy\tidx\tseq\n'
        '50\tTrue\t3.0\t2.0\t[0 2]\tACDE\n',
        encoding='utf-8',
    )


def write_structure_tsv(path):
    path.write_text(
        'record_id\tgen_pct\tcontiguous\tppl\tse\tlength\tptm\t'
        'mean_gen_plddt\tmean_non_gen_plddt\tidx\tseq\terror\n'
        '1\t50\tTrue\t3.0\t2.0\t4\t0.5\t0.7\t0.8\t[0 2]\tACDE\t\n',
        encoding='utf-8',
    )


def write_original_tsv(path):
    path.write_text(
        'id\tseq\tptm\tmean_plddt\noriginal\tACDE\t0.6\t0.7\n',
        encoding='utf-8',
    )


def test_structure_driver_writes_successful_results(tmp_path):
    input_path = tmp_path / 'completion.tsv'
    output_path = tmp_path / 'structure.tsv'
    pdb_dir = tmp_path / 'pdb'
    write_completion_tsv(input_path)

    exit_code = structure_evaluator_driver.main(
        [
            '--input', str(input_path),
            '--output', str(output_path),
            '--pdb-dir', str(pdb_dir),
        ],
        evaluator_factory=lambda model_id, api_key: FakeEvaluator(),
    )

    row = next(csv.DictReader(output_path.open(), delimiter='\t'))
    assert exit_code == 0
    assert row['ptm'] == '0.5'
    assert row['error'] == ''
    assert next(pdb_dir.iterdir()).is_file()


def test_structure_driver_records_errors_without_undefined_metrics(tmp_path):
    class FailingEvaluator:
        def generate_structure(self, **kwargs):
            raise RuntimeError('expected failure')

    output_path = tmp_path / 'failed.tsv'
    successful, failed = structure_evaluator_driver.evaluate_rows(
        [(50.0, True, 3.0, 2.0, [0], 'ACDE')],
        FailingEvaluator(),
        output_path,
        tmp_path / 'pdb',
    )
    row = next(csv.DictReader(output_path.open(), delimiter='\t'))

    assert (successful, failed) == (0, 1)
    assert row['ptm'] == ''
    assert row['error'] == 'RuntimeError: expected failure'


def test_plotting_drivers_parse_inputs_and_call_plot_functions(tmp_path):
    atp_path = tmp_path / 'atp.tsv'
    esm_path = tmp_path / 'esm.tsv'
    original_path = tmp_path / 'original.tsv'
    write_structure_tsv(atp_path)
    write_structure_tsv(esm_path)
    write_original_tsv(original_path)
    combined_calls = []
    length_calls = []
    single_calls = []

    combined_exit = gen_combined_figs_driver.main(
        [
            '--atp', str(atp_path),
            '--esm', str(esm_path),
            '--original', str(original_path),
            '--output-dir', str(tmp_path / 'combined'),
        ],
        combined_figure_fn=lambda *args, **kwargs: combined_calls.append((args, kwargs)),
        length_figure_fn=lambda *args, **kwargs: length_calls.append((args, kwargs)),
    )
    single_exit = gen_figs_single_driver.main(
        [
            '--input', str(atp_path),
            '--original', str(original_path),
            '--output-dir', str(tmp_path / 'single'),
            '--model-label', 'ATP',
        ],
        figure_fn=lambda **kwargs: single_calls.append(kwargs),
    )

    assert combined_exit == single_exit == 0
    assert len(combined_calls) == 1
    assert [call[1]['metric_kind'] for call in length_calls] == ['ptm', 'plddt']
    assert single_calls[0]['model_label'] == 'ATP'
