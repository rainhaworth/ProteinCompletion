"""Inventory public UniProt site annotations and their SIFTS structure mappings.

This is a preflight inventory, not a curated motif-scaffolding benchmark. In
particular, an annotation is not evidence that a structure contains the ligand
or that the selected chain is the biologically relevant assembly.
"""

import argparse
import csv
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FEATURE_TYPES = {
    "Active site",
    "Binding site",
    "DNA binding",
    "Metal binding",
    "Motif",
    "Nucleotide binding",
    "Site",
    "Zinc finger",
}


def exact_position_range(location, sequence_length):
    start = location.get("start", {})
    end = location.get("end", {})
    if start.get("modifier") != "EXACT" or end.get("modifier") != "EXACT":
        return None
    first, last = start.get("value"), end.get("value")
    if not isinstance(first, int) or not isinstance(last, int):
        return None
    if not 1 <= first <= last <= sequence_length:
        return None
    return list(range(first, last + 1))


def parse_sifts_residues(xml_bytes, accession, pdb_id, sequence):
    """Map UniProt 1-based positions to observed author PDB residues per chain."""
    root = ET.fromstring(xml_bytes)
    chains = defaultdict(lambda: defaultdict(list))
    mismatches = []
    for residue in root.iter():
        if residue.tag.rsplit("}", 1)[-1] != "residue":
            continue
        details = [
            child.text for child in residue
            if child.tag.rsplit("}", 1)[-1] == "residueDetail"
        ]
        if "Not_Observed" in details:
            continue
        refs = [
            child.attrib for child in residue
            if child.tag.rsplit("}", 1)[-1] == "crossRefDb"
        ]
        uniprot_refs = [
            ref for ref in refs
            if ref.get("dbSource") == "UniProt"
            and ref.get("dbAccessionId") == accession
        ]
        pdb_refs = [
            ref for ref in refs
            if ref.get("dbSource") == "PDB"
            and ref.get("dbAccessionId", "").upper() == pdb_id.upper()
            and ref.get("dbResNum") not in (None, "null")
        ]
        for uni in uniprot_refs:
            try:
                position = int(uni["dbResNum"])
            except (KeyError, ValueError):
                continue
            if not 1 <= position <= len(sequence):
                mismatches.append({"uniprot_position": position, "reason": "out of range"})
                continue
            residue_letter = uni.get("dbResName")
            if residue_letter and residue_letter != sequence[position - 1]:
                mismatches.append({
                    "uniprot_position": position,
                    "panel_residue": sequence[position - 1],
                    "sifts_residue": residue_letter,
                })
            for pdb in pdb_refs:
                chain = pdb.get("dbChainId")
                if chain:
                    site = {"pdb_author_residue": pdb["dbResNum"], "pdb_resname": pdb.get("dbResName")}
                    if site not in chains[chain][position]:
                        chains[chain][position].append(site)
    return {
        chain: {str(pos): sites for pos, sites in sorted(positions.items())}
        for chain, positions in sorted(chains.items())
    }, mismatches


def candidate_features(uniprot_record, sequence, chains):
    counts = Counter(feature.get("type", "unknown") for feature in uniprot_record.get("features", []))
    selected = []
    for feature in uniprot_record.get("features", []):
        if feature.get("type") not in FEATURE_TYPES:
            continue
        positions = exact_position_range(feature.get("location", {}), len(sequence))
        if positions is None:
            continue
        mapping = {
            chain: {str(pos): sites[str(pos)] for pos in positions if str(pos) in sites}
            for chain, sites in chains.items()
        }
        selected.append({
            "type": feature["type"],
            "description": feature.get("description"),
            "ligand": feature.get("ligand"),
            "positions_uniprot_1based": positions,
            "evidence_codes": sorted({
                ev["evidenceCode"] for ev in feature.get("evidences", [])
                if "evidenceCode" in ev
            }),
            "evidences": feature.get("evidences", []),
            "mapped_pdb_residues_by_chain": mapping,
            "fully_observed_chains": [
                chain for chain, sites in mapping.items() if len(sites) == len(positions)
            ],
        })
    return selected, dict(sorted(counts.items()))


def fetch_cached(url, path, delay=0.2):
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ProteinCompletion-motif-inventory/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            path.write_bytes(content)
            time.sleep(delay)
            return content
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not fetch {url}")


def inventory_record(panel_entry, cache_dir):
    accession = panel_entry["accession"]
    pdb_id = panel_entry["reference_pdb"].lower()
    sequence = panel_entry["sequence"]
    uni_url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    sifts_url = f"https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id}.xml.gz"
    uni = json.loads(fetch_cached(uni_url, cache_dir / "uniprot" / f"{accession}.json"))
    xml = gzip.decompress(fetch_cached(sifts_url, cache_dir / "sifts" / f"{pdb_id}.xml.gz"))
    uni_sequence = uni.get("sequence", {}).get("value")
    sequence_matches = uni_sequence == sequence
    chains, mismatches = parse_sifts_residues(xml, accession, pdb_id, sequence)
    features, feature_counts = candidate_features(uni, sequence, chains)
    chain_coverage = {chain: len(mapping) for chain, mapping in chains.items()}
    max_observed_fraction = max(chain_coverage.values(), default=0) / len(sequence)
    flags = []
    if not sequence_matches:
        flags.append("panel_sequence_differs_from_current_uniprot")
    if not chains:
        flags.append("no_observed_sifts_mapping_to_panel_accession")
    if max_observed_fraction < 0.9:
        flags.append("observed_reference_mapping_below_90_percent")
    if mismatches:
        flags.append("sifts_residue_disagrees_with_panel_sequence")
    if not features:
        flags.append("no_candidate_site_annotation")
    elif not any(feature["fully_observed_chains"] for feature in features):
        flags.append("no_fully_observed_candidate_site")
    if len(chains) > 1:
        flags.append("multiple_reference_chains_require_selection")
    return {
        **{key: panel_entry[key] for key in ("accession", "length", "length_bin", "family", "reference_pdb", "ca_coverage")},
        "sequence_matches_current_uniprot": sequence_matches,
        "uniprot_entry_type": uni.get("entryType"),
        "uniprot_annotation_update_date": uni.get("entryAudit", {}).get("lastAnnotationUpdateDate"),
        "uniprot_feature_counts": feature_counts,
        "sifts_observed_residue_count_by_chain": chain_coverage,
        "max_observed_uniprot_fraction": round(max_observed_fraction, 6),
        "sifts_uniprot_residue_mismatches": mismatches,
        "candidate_features": features,
        "preflight_flags": flags,
        "source_urls": {"uniprot": uni_url, "sifts": sifts_url},
    }


def build_inventory(panel_path, output_path, cache_dir, limit=None):
    panel_bytes = panel_path.read_bytes()
    panel = json.loads(panel_bytes)
    if limit is not None:
        panel = panel[:limit]
    records, errors = [], []
    for number, entry in enumerate(panel, start=1):
        try:
            record = inventory_record(entry, cache_dir)
            records.append(record)
            print(f"{number}/{len(panel)} {entry['accession']} {len(record['candidate_features'])} candidate features", flush=True)
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            errors.append({"accession": entry.get("accession"), "pdb": entry.get("reference_pdb"), "error": str(error)})
            print(f"{number}/{len(panel)} {entry.get('accession')} FAILED: {error}", flush=True)
    result = {
        "purpose": "Candidate functional-annotation inventory only; motifs and structures require biological review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "requested_records": len(panel),
        "completed_records": len(records),
        "summary": {
            "matching_current_uniprot_sequence": sum(record["sequence_matches_current_uniprot"] for record in records),
            "proteins_with_candidate_feature": sum(bool(record["candidate_features"]) for record in records),
            "proteins_with_fully_observed_candidate_feature": sum(
                any(feature["fully_observed_chains"] for feature in record["candidate_features"])
                for record in records
            ),
            "candidate_features": sum(len(record["candidate_features"]) for record in records),
            "fully_observed_candidate_features": sum(
                bool(feature["fully_observed_chains"])
                for record in records for feature in record["candidate_features"]
            ),
            "proteins_with_multiple_observed_chains": sum(
                len(record["sifts_observed_residue_count_by_chain"]) > 1 for record in records
            ),
            "proteins_without_observed_mapping": sum(
                not record["sifts_observed_residue_count_by_chain"] for record in records
            ),
            "proteins_below_90_percent_observed_mapping": sum(
                record["max_observed_uniprot_fraction"] < 0.9 for record in records
            ),
        },
        "errors": errors,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_review_table(records, path):
    """Write one row per annotation for manual motif and chain selection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "accession", "reference_pdb", "feature_index", "type", "description",
        "ligand", "uniprot_positions_1based", "evidence_codes",
        "fully_observed_chains", "mapped_residue_count_by_chain", "preflight_flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            for index, feature in enumerate(record["candidate_features"]):
                writer.writerow({
                    "accession": record["accession"],
                    "reference_pdb": record["reference_pdb"],
                    "feature_index": index,
                    "type": feature["type"],
                    "description": feature["description"],
                    "ligand": json.dumps(feature["ligand"]) if feature["ligand"] else "",
                    "uniprot_positions_1based": ",".join(map(str, feature["positions_uniprot_1based"])),
                    "evidence_codes": ",".join(feature["evidence_codes"]),
                    "fully_observed_chains": ",".join(feature["fully_observed_chains"]),
                    "mapped_residue_count_by_chain": json.dumps({
                        chain: len(sites)
                        for chain, sites in feature["mapped_pdb_residues_by_chain"].items()
                    }),
                    "preflight_flags": ",".join(record["preflight_flags"]),
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/candidate_motif_inventory.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/public_annotation_cache"))
    parser.add_argument("--review-table", type=Path, default=Path("data/candidate_motif_review.csv"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = build_inventory(args.panel, args.output, args.cache_dir, args.limit)
    write_review_table(result["records"], args.review_table)
    print(f"Inventory saved to {args.output}; {result['completed_records']} complete, {len(result['errors'])} errors")
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
