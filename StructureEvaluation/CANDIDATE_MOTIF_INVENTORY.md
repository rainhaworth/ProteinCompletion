# Candidate motif inventory

This is a screening step for the 60-protein candidate panel, not a final
motif-scaffolding test set. It adds UniProt functional-site annotations and
SIFTS residue mappings to reference PDB chains. It does not choose a biological
assembly, combine individual annotated residues into a motif, or establish
that the cited PDB structure contains a relevant ligand or partner.

The input panel used here has SHA-256
`ca6608152c3931b72de43df6b228ac4dc2194b58243c1a4da5383060dc893ee2`.
To regenerate the local files from that panel:

```powershell
python -m StructureEvaluation.candidate_motif_panel path/to/panel.json
python -m StructureEvaluation.atp_schedule_diagnostic --length 10 --motif 3 6
```

The first command writes `data/candidate_motif_inventory.json` and
`data/candidate_motif_review.csv`. Public UniProt and SIFTS responses are
cached under `data/public_annotation_cache`. These outputs are ignored by Git.
The CSV is intended for manual review; the JSON retains exact UniProt positions,
author PDB residue numbers, chain IDs, evidence, and flags. A fully mapped
annotation is still only a candidate motif component.

For this panel, all 60 sequences matched current UniProt records. There were
272 candidate annotations on 43 proteins, and 260 annotations on 42 proteins
mapped completely to at least one observed PDB chain. Of the 60 proteins, 39
had multiple mapped chains. Most annotations were individual binding-site
residues, so the annotation count is not a count of distinct functional motifs.
`P0AEX9` had no observed SIFTS residues in its cited structure, `9BOI`, despite
the panel's stated high C-alpha coverage. That reference needs manual review
before use.

The ATP diagnostic checks predictor-row visibility against the packed training
mask. Parallel frontier expansion matches direct row visibility in exhaustive
small-sequence tests. Serial generation can use different visible context and
can produce a predictor-target pair absent from the packed training targets.
Even parallel generation does not necessarily reproduce the same multilayer
hidden states, because other rows can have narrower context in packed training.
The diagnostic does not measure sequence quality or replace a comparison with
the corrected checkpoint.
