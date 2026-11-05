# Wild-type AlphaFold 3 pilot

The first run uses the 246-residue A0A0H2XJL0 sequence and the 9C0L catalytic
triad as a reference. This checks AF3 output numbering and the geometry-scoring
path. It is not a model comparison or a benchmark result.

Generate the input from the candidate panel:

```powershell
python -m StructureEvaluation.prepare_af3_pilot path/to/panel.json
```

This writes `data/af3_wt_pilot.json`. It requests one protein chain, one fixed
seed, no MSA, and no structural templates. For the eventual automatic-MSA
version, add `--auto-msa` and set `--output` to
`data/af3_wt_pilot_auto_msa.json`. The latter will require the full AF3 database
search and is not needed to check the output parser.

The input can be checked against the installed AlphaFold 3 parser without a
GPU. Copy `data/af3_wt_pilot.json` to Zaratan, load `alphafold3/3.0.1`, and
run this on a login node:

```bash
python -c 'from pathlib import Path; from alphafold3.common import folding_input; job = folding_input.Input.from_json(Path("af3_wt_pilot.json").read_text()); assert len(job.chains) == 1 and job.rng_seeds == (1,); print("Input parsed successfully")'
```

The 246-residue input passed this check with the installed Zaratan 3.0.1 module
on 2026-09-17, returning `a0a0h2xjl0_wt_msa_free 1 (1,)`. This establishes
input compatibility, not folding behavior. The local environment does not have
AlphaFold 3 installed.

On Zaratan, first confirm an active Slurm account, access to a GPU allocation,
and the location of model parameters. The installed `alphafold3/3.0.1` module
sets `ALPHAFOLD_DATABASE_DIR`, but its default `--model_dir` points to
`$HOME/models`, which was absent in the prior inspection. The model parameters
must be obtained and used under their terms. Do not run inference on a login
node.

After copying the JSON to an allocated GPU job, the command is:

```bash
module load alphafold3/3.0.1
run_alphafold.py \
  --json_path=/path/to/af3_wt_pilot.json \
  --model_dir=/path/to/af3/model/parameters \
  --db_dir="$ALPHAFOLD_DATABASE_DIR" \
  --output_dir=/path/to/pilot/output \
  --num_diffusion_samples=1
```

The job name is `a0a0h2xjl0_wt_msa_free`. In its output directory, copy the
top-ranked `a0a0h2xjl0_wt_msa_free_model.cif` and both top-ranked confidence
JSON files locally. Then run:

```powershell
python -m StructureEvaluation.score_af3_pilot path/to/a0a0h2xjl0_wt_msa_free_model.cif --input-json data/af3_wt_pilot.json --confidence-dir path/to/a0a0h2xjl0_wt_msa_free
```

The scorer compares UniProt positions 93, 190, and 220 with SIFTS-mapped author
residues 93, 190, and 220 of 9C0L chain A. It requires every C-alpha residue
in the predicted chain to match the input sequence at the expected label
sequence number before reporting motif C-alpha RMSD and pair-distance error.
Inspect the AF3 mmCIF's label sequence numbering before treating the score as
valid. The confidence summary reports pTM, chain pTM, ranking score, and mean
atom pLDDT for the chain. These are model confidence estimates, not measured
folding accuracy or evidence of binding.
The deposited 9C0L structure does not use UniProt numbering in its label
sequence IDs, so it cannot stand in for a full-length AF3 prediction in this
check. This smoke test has only been exercised with synthetic prediction files;
the file conventions still need to be checked against a real AF3 result.

For checkpoint-independent preparation of the later sequence-generation tests:

```powershell
python -m StructureEvaluation.pilot_generation_manifest path/to/panel.json
```

This writes `data/pilot_generation_manifest.json` with two fixed generation
seeds per pilot protein. Each row records the full original sequence, exact
zero-based visible positions, attempt number, and seed. Both sequence models
can use the same rows. The manifest is provisional and is not an approved test
set.

`run_pilot_generation` consumes the manifest directly so fragmented motifs are
not split into separate examples by the legacy binding-site TSV reader. It also
recreates the current UniRef tokenizer layout, verifies the checkpoint hash,
checks motif preservation, and writes each completed attempt atomically.

The 80,000-step checkpoints received on 2026-10-10 have these SHA-256 hashes:

- ESM: `d507f9abf5f0cb87265bc321715c7921146a31bef276c307c816b42fa70fcfb5`
- ATP: `656b77646ba607892f6fc4685d46b70ab3c0db48089a55851f9d565d512f15a3`

Run one matched smoke case on a GPU before launching all six rows:

```bash
python -m StructureEvaluation.run_pilot_generation \
  --manifest data/pilot_generation_manifest.json \
  --weights /path/to/train-esm-step80000.pt \
  --checkpoint-sha256 d507f9abf5f0cb87265bc321715c7921146a31bef276c307c816b42fa70fcfb5 \
  --model-type esm \
  --accession A0A0H2XJL0 \
  --attempt-index 0 \
  --output results/pilot-esm-smoke.json

python -m StructureEvaluation.run_pilot_generation \
  --manifest data/pilot_generation_manifest.json \
  --weights /path/to/train-atp-step80000.pt \
  --checkpoint-sha256 656b77646ba607892f6fc4685d46b70ab3c0db48089a55851f9d565d512f15a3 \
  --model-type atp \
  --atp-length-mode fixed \
  --accession A0A0H2XJL0 \
  --attempt-index 0 \
  --output results/pilot-atp-smoke.json
```

If both records are complete and preserve their motifs, omit the accession and
attempt filters to generate all six attempts for each model. The runner defaults
to nucleus sampling with `p=0.95`, the medium-large model configuration, and the
seed recorded in each manifest row.

The matched ATP-versus-ESM comparison should use `--atp-length-mode fixed`, so
both models receive the same target sequence length. ATP's default `free` mode
instead crops to the motif span and asks the model to generate both terminal
tokens and the sequence length. That is a separate variable-length generation
task and should be reported separately rather than compared directly with the
fixed-length ESM completion.
