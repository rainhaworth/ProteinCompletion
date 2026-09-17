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

The job name is `a0a0h2xjl0_wt_msa_free`. In its output directory, take the
top-ranked `a0a0h2xjl0_wt_msa_free_model.cif` and copy it locally. Then run:

```powershell
python -m StructureEvaluation.score_af3_pilot path/to/a0a0h2xjl0_wt_msa_free_model.cif
```

The scorer compares UniProt positions 93, 190, and 220 with SIFTS-mapped author
residues 93, 190, and 220 of 9C0L chain A. It checks their amino-acid
identities before reporting motif C-alpha RMSD and pair-distance error. Inspect
the AF3 mmCIF's label sequence numbering before treating the score as valid.
The deposited 9C0L structure does not use UniProt numbering in its label
sequence IDs, so it cannot stand in for a full-length AF3 prediction in this
check. The AF3 confidence files should be retained with the structure for later
evaluation; this smoke check does not interpret them.
