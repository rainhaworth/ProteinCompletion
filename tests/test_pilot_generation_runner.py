from types import SimpleNamespace

import pytest
import torch

from StructureEvaluation.run_pilot_generation import (
    generate_case,
    make_uniref_tokenizer,
    prepare_case,
    select_rows,
    sha256_file,
    write_payload,
)


def example_row():
    return {
        "accession": "EXAMPLE",
        "reference_pdb": "1ABC",
        "original_sequence": "ACDEFG",
        "visible_positions_0based": [1, 4],
        "attempt_index": 0,
        "generation_seed": 17,
    }


def test_current_uniref_tokenizer_layout_matches_training_pipeline():
    tokenizer = make_uniref_tokenizer()
    assert tokenizer.token_to_id("<mask>") == 4
    assert tokenizer.token_to_id("A") == 5
    assert tokenizer.token_to_id("Z") == 30
    assert tokenizer.token_to_id("<unk>") == 31
    assert tokenizer.encode("ACDEFG").ids == [5, 7, 8, 9, 10, 11]


def test_prepare_case_preserves_full_esm_context_and_crops_atp_to_motif_span():
    tokenizer = make_uniref_tokenizer()
    row = example_row()
    esm_sequence, esm_visible, esm_original, esm_motif = prepare_case(
        row, tokenizer, "esm", torch.device("cpu")
    )
    atp_sequence, atp_visible, atp_original, atp_motif = prepare_case(
        row, tokenizer, "atp", torch.device("cpu")
    )
    fixed_atp_sequence, fixed_atp_visible, _, fixed_atp_motif = prepare_case(
        row,
        tokenizer,
        "atp",
        torch.device("cpu"),
        atp_length_mode="fixed",
    )

    assert esm_sequence.shape == (1, 6)
    assert esm_visible.tolist() == [1, 4]
    assert atp_sequence.shape == (1, 4)
    assert atp_visible.tolist() == [0, 3]
    assert fixed_atp_sequence.shape == (1, 6)
    assert fixed_atp_visible.tolist() == [1, 4]
    assert esm_original.tolist() == atp_original.tolist() == [1, 4]
    assert esm_motif.tolist() == atp_motif.tolist() == fixed_atp_motif.tolist()


def test_fixed_length_atp_fills_manifest_window_without_predicting_terminals():
    tokenizer = make_uniref_tokenizer()
    row = example_row()
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=32, n_ctx=16))
    terminal_flags = []

    def reveal_original_token(
        model,
        sequence,
        visible,
        device,
        invalid_ids,
        sample_fn,
        bos_id,
        eos_id,
        predict_terminals,
    ):
        terminal_flags.append(predict_terminals)
        hidden = sorted(set(range(sequence.size(1))) - set(visible.tolist()))
        if not hidden:
            return None, None
        position = hidden[0]
        return sequence[0, position], position

    result = generate_case(
        model,
        row,
        "atp",
        tokenizer,
        torch.device("cpu"),
        sample_fn=lambda values: (torch.tensor(0), torch.tensor(0)),
        generation_step=reveal_original_token,
        atp_length_mode="fixed",
    )

    assert result["complete"] is True
    assert result["length_mode"] == "fixed"
    assert result["sequence"] == row["original_sequence"]
    assert result["generated_length"] == result["original_length"] == 6
    assert result["generated_visible_positions_0based"] == [1, 4]
    assert terminal_flags and not any(terminal_flags)


def test_select_rows_rejects_bad_or_empty_filters():
    manifest = {"schema_version": 1, "rows": [example_row()]}
    assert select_rows(manifest, "EXAMPLE", 0) == manifest["rows"]
    with pytest.raises(ValueError, match="No pilot rows"):
        select_rows(manifest, "OTHER", None)
    with pytest.raises(ValueError, match="schema-version 1"):
        select_rows({"rows": []})


def test_esm_case_generation_is_complete_and_preserves_fragmented_motif():
    tokenizer = make_uniref_tokenizer()
    row = example_row()
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=32, n_ctx=16))

    def reveal_original_token(
        model,
        sequence,
        visible,
        device,
        invalid_ids,
        sample_fn,
        mask_id,
    ):
        hidden = sorted(set(range(sequence.size(1))) - set(visible.tolist()))
        if not hidden:
            return None, None
        position = hidden[0]
        return sequence[0, position], position

    result = generate_case(
        model,
        row,
        "esm",
        tokenizer,
        torch.device("cpu"),
        sample_fn=lambda values: (torch.tensor(0), torch.tensor(0)),
        generation_step=reveal_original_token,
    )
    assert result["complete"] is True
    assert result["sequence"] == row["original_sequence"]
    assert result["original_visible_positions_0based"] == [1, 4]
    assert result["generated_visible_positions_0based"] == [1, 4]
    assert result["motif_sequence"] == "CF"


def test_generation_rejects_invalid_manifest_sequences():
    tokenizer = make_uniref_tokenizer()
    row = example_row()
    row["original_sequence"] = "ACD?FG"
    with pytest.raises(ValueError, match="invalid sequence"):
        prepare_case(row, tokenizer, "esm", torch.device("cpu"))


def test_checkpoint_hash_and_atomic_payload_write(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert sha256_file(checkpoint) == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )

    output = tmp_path / "result.json"
    write_payload(output, {"results": [{"complete": True}]})
    assert output.exists()
    assert not output.with_suffix(".json.tmp").exists()
