"""Generate the fixed motif-completion pilot from a manifest."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from generate import prepare_generation_output
from utils.evaluation import normalize_model_type
from utils.generation import (
    gen_step_atp,
    gen_step_esmlike,
    make_sample_fn,
    resolve_terminal_ids,
    update_generation_state,
)
from utils.model_bidirectional import BidirectionalCausalLM
from utils.model_esmlike import ESMlikeLM
from utils.utils import load_model_checkpoint, set_env, set_seed


VALID_AAS = "ACDEFGHIKLMNPQRSTVWY"


def make_uniref_tokenizer():
    """Recreate the tokenizer used by the current UniRef packing pipeline."""
    tokens = ["<pad>", "<bos>", "<eos>", "<sep>", "<mask>"] + list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    vocab = {token: index for index, token in enumerate(tokens)}
    vocab["<unk>"] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, "<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    return tokenizer


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select_rows(manifest, accession=None, attempt_index=None):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("rows"), list):
        raise ValueError("Expected a schema-version 1 pilot manifest with rows")
    rows = manifest["rows"]
    if accession is not None:
        rows = [row for row in rows if row.get("accession") == accession]
    if attempt_index is not None:
        rows = [row for row in rows if row.get("attempt_index") == attempt_index]
    if not rows:
        raise ValueError("No pilot rows matched the requested filters")
    return rows


def prepare_case(
    row,
    tokenizer,
    model_type,
    device,
    atp_length_mode="free",
):
    sequence = row["original_sequence"]
    if not sequence or any(residue not in VALID_AAS for residue in sequence):
        raise ValueError(f"{row.get('accession', 'Pilot row')} has an invalid sequence")
    token_ids = tokenizer.encode(sequence).ids
    if len(token_ids) != len(sequence):
        raise ValueError("Pilot generation requires one token per residue")

    visible = torch.as_tensor(
        row["visible_positions_0based"], dtype=torch.long, device=device
    ).reshape(-1)
    visible = torch.unique(visible, sorted=True)
    if visible.numel() == 0 or visible[0] < 0 or visible[-1] >= len(sequence):
        raise ValueError("Pilot visible positions are empty or outside the sequence")
    if visible.numel() == len(sequence):
        raise ValueError("Pilot row must contain at least one hidden position")

    original_visible = visible.clone()
    if atp_length_mode not in {"free", "fixed"}:
        raise ValueError("ATP length mode must be 'free' or 'fixed'")
    if model_type == "atp" and atp_length_mode == "free":
        crop_start = int(visible[0])
        crop_end = int(visible[-1]) + 1
        token_ids = token_ids[crop_start:crop_end]
        visible = visible - crop_start

    sequence_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
    motif_tokens = sequence_tensor[0, visible].clone()
    return sequence_tensor, visible, original_visible, motif_tokens


def generate_case(
    model,
    row,
    model_type,
    tokenizer,
    device,
    sample_fn,
    max_steps=None,
    generation_step=None,
    atp_length_mode="free",
):
    sequence, visible, original_visible, motif_tokens = prepare_case(
        row,
        tokenizer,
        model_type,
        device,
        atp_length_mode=atp_length_mode,
    )
    motif_indices = visible.clone()
    bos_id, eos_id = resolve_terminal_ids(tokenizer)
    mask_id = tokenizer.token_to_id("<mask>")
    valid_ids = set(tokenizer.encode(VALID_AAS).ids)
    invalid_ids = [
        token_id for token_id in range(model.config.vocab_size) if token_id not in valid_ids
    ]

    if generation_step is None:
        generation_step = gen_step_atp if model_type == "atp" else gen_step_esmlike
    step_limit = max_steps or model.config.n_ctx
    generated_steps = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(step_limit):
            new_token, new_position = generation_step(
                model,
                sequence,
                visible,
                device,
                invalid_ids,
                sample_fn=sample_fn,
                **(
                    {
                        "bos_id": bos_id,
                        "eos_id": eos_id,
                        "predict_terminals": atp_length_mode == "free",
                    }
                    if model_type == "atp"
                    else {"mask_id": mask_id}
                ),
            )
            if new_token is None:
                break
            if int(torch.as_tensor(new_position).item()) == -1:
                motif_indices = motif_indices + 1
            sequence, visible = update_generation_state(
                sequence, visible, new_token, new_position
            )
            generated_steps += 1
            if model_type == "atp" and sequence.size(1) >= model.config.n_ctx:
                break

    all_visible = visible.numel() == sequence.size(1)
    has_terminals = (
        sequence.size(1) >= 2
        and sequence[0, 0].item() == bos_id
        and sequence[0, -1].item() == eos_id
    )
    complete = (
        all_visible
        if model_type == "esm" or atp_length_mode == "fixed"
        else all_visible and has_terminals
    )
    generated_sequence, output_motif, output_visible = prepare_generation_output(
        sequence,
        visible,
        motif_indices,
        tokenizer,
        complete,
        bos_id,
        eos_id,
    )
    if complete:
        generated_motif_tokens = torch.tensor(
            tokenizer.encode(generated_sequence).ids, dtype=torch.long, device=device
        )[output_motif]
        if not torch.equal(generated_motif_tokens, motif_tokens):
            raise RuntimeError("Generation changed at least one visible motif residue")
        if any(residue not in VALID_AAS for residue in generated_sequence):
            raise RuntimeError("Completed sequence contains a non-amino-acid token")

    return {
        "accession": row["accession"],
        "reference_pdb": row.get("reference_pdb"),
        "attempt_index": row["attempt_index"],
        "generation_seed": row["generation_seed"],
        "model_type": model_type,
        "length_mode": atp_length_mode if model_type == "atp" else "fixed",
        "complete": complete,
        "original_length": len(row["original_sequence"]),
        "generated_length": len(generated_sequence),
        "generated_steps": generated_steps,
        "original_visible_positions_0based": original_visible.detach().cpu().tolist(),
        "generated_visible_positions_0based": output_motif.detach().cpu().tolist(),
        "all_visible_positions_0based": output_visible.detach().cpu().tolist(),
        "motif_sequence": "".join(
            row["original_sequence"][index]
            for index in original_visible.detach().cpu().tolist()
        ),
        "sequence": generated_sequence,
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--model-type", choices=["atp", "esm"], required=True)
    parser.add_argument("--config", type=Path, default=Path("config-medlarge.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample", choices=["nucleus", "greedy"], default="nucleus")
    parser.add_argument("--p", type=float, default=0.95)
    parser.add_argument("--accession")
    parser.add_argument("--attempt-index", type=int)
    parser.add_argument(
        "--atp-length-mode",
        choices=["free", "fixed"],
        default="free",
        help=(
            "For ATP, either generate sequence terminals and length (free) or "
            "fill the manifest's fixed sequence length without terminal prediction "
            "(fixed). ESM generation is always fixed-length."
        ),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 < args.p <= 1:
        raise ValueError("--p must be greater than zero and at most one")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    checkpoint_sha256 = sha256_file(args.weights)
    if (
        args.checkpoint_sha256 is not None
        and checkpoint_sha256.lower() != args.checkpoint_sha256.lower()
    ):
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, "
            f"found {checkpoint_sha256}"
        )

    set_env()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model_type = normalize_model_type(args.model_type)
    model_class = BidirectionalCausalLM if model_type == "atp" else ESMlikeLM
    model = load_model_checkpoint(
        model_class, str(args.config), device, str(args.weights)
    ).eval()
    tokenizer = make_uniref_tokenizer()
    if tokenizer.token_to_id("<mask>") != 4 or tokenizer.token_to_id("A") != 5:
        raise RuntimeError("Unexpected current UniRef tokenizer layout")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = select_rows(manifest, args.accession, args.attempt_index)
    sample_fn = make_sample_fn(args.sample, args.p)
    payload = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "model_type": model_type,
        "checkpoint": args.weights.name,
        "checkpoint_sha256": checkpoint_sha256,
        "config": args.config.name,
        "tokenizer_layout": {
            "mask_id": tokenizer.token_to_id("<mask>"),
            "a_id": tokenizer.token_to_id("A"),
            "unknown_id": tokenizer.token_to_id("<unk>"),
        },
        "results": [],
    }
    for row in rows:
        set_seed(row["generation_seed"], deterministic=True)
        result = generate_case(
            model,
            row,
            model_type,
            tokenizer,
            device,
            sample_fn,
            max_steps=args.max_steps,
            atp_length_mode=args.atp_length_mode,
        )
        result.update(
            {
                "checkpoint": args.weights.name,
                "config": args.config.name,
                "sampling_method": args.sample,
                "nucleus_p": args.p if args.sample == "nucleus" else None,
            }
        )
        payload["results"].append(result)
        write_payload(args.output, payload)
        print(
            f"{model_type} {result['accession']} attempt {result['attempt_index']}: "
            f"complete={result['complete']} length={result['generated_length']} "
            f"time={result['elapsed_seconds']:.1f}s"
        )

    print(f"Saved {len(payload['results'])} results to {args.output}")


if __name__ == "__main__":
    main()
