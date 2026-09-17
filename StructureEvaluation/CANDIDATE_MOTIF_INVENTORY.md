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
the panel's stated C-alpha coverage of 0.917. The deposited construct is a
fusion containing several proteins; its P0AEX9 portion is unobserved. This
contradicts the panel manifest's claim that coverage measures modeled C-alpha
residues. Other entries also have lower observed SIFTS coverage than the panel
reports. The reference-structure screen needs correction before using this
panel for structure-based scores.
On a per-chain basis, 12 of 60 proteins have fewer than 90% of their UniProt
positions mapped to observed residues in their nominated PDB entry. This is a
mapping-based check, not a direct recount of C-alpha atoms from coordinates.

The ATP diagnostic checks predictor-row visibility against the packed training
mask. Parallel frontier expansion matches direct row visibility in exhaustive
small-sequence tests. Serial generation can use different visible context and
can produce a predictor-target pair absent from the packed training targets.
Even parallel generation does not necessarily reproduce the same multilayer
hidden states, because other rows can have narrower context in packed training.
The diagnostic does not measure sequence quality or replace a comparison with
the corrected checkpoint.

## Reference-geometry pilots

`pilot_motifs.json` contains three provisional residue sets for testing the
structure-scoring code. They are not the final evaluation set.

| UniProt / PDB | Assembly and author chain | UniProt positions | Why this is a useful pilot | Limit |
| --- | --- | --- | --- | --- |
| A0A0H2XJL0 / [9C0L](https://www.rcsb.org/structure/9C0L) | 1 / A | 93, 190, 220 | Catalytic triad, nearly complete monomer structure | Apo structure; site annotations are inferred |
| I6YFP0 / [9DRN](https://www.rcsb.org/structure/9DRN) | 1 / B | 38, 39, 63, 67, 138 | Biotin-site residues near a bound biotin analog | Analog rather than biotin; author chain B is label chain A |
| Q9F663 / [9F0V](https://www.rcsb.org/structure/9F0V) | 1 / A | 69, 72, 129, 165 | Catalytic residues in an inhibitor-bound monomer | Inhibitor rather than beta-lactam substrate; author residue IDs are shifted by one |

To run the local preflight, download the three public mmCIF files and check
them against the recorded residue mappings:

```powershell
New-Item -ItemType Directory -Path data/reference_cif -Force | Out-Null
foreach ($pdb in @('9C0L', '9DRN', '9F0V')) {
    Invoke-WebRequest "https://files.rcsb.org/download/$pdb.cif" -OutFile "data/reference_cif/$pdb.cif"
}
python -m StructureEvaluation.reference_geometry data/reference_cif
```

The command writes `data/pilot_reference_geometry.json`. All three reference
motifs have the expected residue identities and C-alpha coordinates in their
selected assemblies. Self-comparison produces effectively zero aligned motif
C-alpha RMSD. In 9DRN, nearest heavy-atom distances from the five selected
residues to the bound analog are 2.64–3.30 Å. Comparing its two deposited
protein chains gives a 0.182 Å motif C-alpha RMSD. These checks validate
numbering, chain selection, and geometry code; they do not measure prediction
quality or binding. The code can compare a future full-length AF3 mmCIF file
against these references, but AF3 output numbering must be checked first.
The preflight reads deposited coordinates and checks assembly membership; it
does not build transformed multichain assemblies or score inter-chain contacts.

The remaining panel entries have not been biologically curated. In particular,
P0AEX9/9BOI should not be used as a structural reference in its present form.
The [9J2L](https://www.rcsb.org/structure/9J2L) entry contains Fe(II), while
the mapped UniProt site annotations concern Zn(II), so those site labels
should not be treated as evidence for the observed metal environment.
