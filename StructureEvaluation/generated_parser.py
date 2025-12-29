import re

def _parse_idx_cell(s):
    """
    Convert the 'idx' field into a list of integers.
    Handles both space-separated and comma-separated formats.
    Handles multiline idx fields
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if not s:
        return []
    parts = s.split(",") if "," in s else s.split()
    return [int(x) for x in parts if x]

def parse_tsv(tsv_path):
    """
    Parse a TSV file containing protein sequence data and extract:
      - generated % (float)
      - contiguous (bool)
      - PPL (float)
      - SE (float)
      - idx (list[int])
      - seq (str)

    Returns:
        List of tuples:
        (gen_pct, contiguous, ppl, se, idx, seq)
    """
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        buf = ""
        for raw in f:
            line = raw.rstrip("\n")
            buf = (buf + "\n" + line) if buf else line

            # Wait until we have 6 fields (5 tabs)
            if buf.count("\t") < 5:
                continue

            gen_str, contig_str, ppl_str, se_str, idx_str, seq_str = buf.split("\t", 5)
            buf = ""  # reset

            # Skip header/non-numeric first field
            try:
                gen_pct = float(gen_str.strip())
            except ValueError:
                continue

            contiguous = contig_str.strip().lower() == "true"

            s = ppl_str.strip()
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                t = re.split(r'[\s,]+', inner)
                ppl = (float(t[0]) + float(t[1])) / 2.0
            else:
                ppl = float(s)

            se = float(se_str.strip())
            idx = _parse_idx_cell(idx_str)
            seq = seq_str.strip()

            rows.append((gen_pct, contiguous, ppl, se, idx, seq))

        # Flush if file ends with a complete record buffered
        if buf and buf.count("\t") >= 5:
            gen_str, contig_str, ppl_str, se_str, idx_str, seq_str = buf.split("\t", 5)
            try:
                gen_pct = float(gen_str.strip())
            except ValueError:
                return rows  # leftover was header/invalid

            contiguous = contig_str.strip().lower() == "true"
            s = ppl_str.strip()
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                t = re.split(r'[\s,]+', inner)
                ppl = (float(t[0]) + float(t[1])) / 2.0
            else:
                ppl = float(s)
            se = float(se_str.strip())
            idx = _parse_idx_cell(idx_str)
            seq = seq_str.strip()
            rows.append((gen_pct, contiguous, ppl, se, idx, seq))

    return rows

def _parse_ppl_cell(s):
    """Accept 'x.y' or '[for back]' (spaces or commas) and return a float (mean if pair)."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        toks = re.split(r'[\s,]+', inner)
        if len(toks) >= 2:
            return (float(toks[0]) + float(toks[1])) / 2.0
        elif len(toks) == 1:
            return float(toks[0])
        else:
            raise ValueError("Empty bracketed PPL")
    return float(s)

def parse_rich_tsv(tsv_path):
    """
    Parse TSV with header (order may vary). Expected columns include:
      record_id, gen_pct (or 'generated'), contiguous, ppl, se, length,
      ptm, mean_gen_plddt, mean_non_gen_plddt, idx, seq

    - Handles multi-line records (e.g., idx spans lines but has no tabs)
    - Returns: list of dicts with keys:
        record_id, gen_pct, contiguous, ppl, se, length, ptm,
        mean_gen_plddt, mean_non_gen_plddt, idx, seq
    """
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return rows

        header = [h.strip() for h in header_line.rstrip("\n").split("\t")]
        n_fields = len(header)

        # Map header names (case-insensitive) to canonical keys
        canon = {
            "record_id": "record_id",
            "gen_pct": "gen_pct",
            "generated": "gen_pct",  # alias
            "contiguous": "contiguous",
            "ppl": "ppl",
            "se": "se",
            "length": "length",
            "ptm": "ptm",
            "mean_gen_plddt": "mean_gen_plddt",
            "mean_non_gen_plddt": "mean_non_gen_plddt",
            "idx": "idx",
            "seq": "seq",
        }

        idx_to_key = {}
        for i, name in enumerate(header):
            k = canon.get(name.strip().lower())
            if k:
                idx_to_key[i] = k

        buf = ""
        def _emit(parts):
            raw_row = {}
            for i, val in enumerate(parts):
                key = idx_to_key.get(i)
                if key:
                    raw_row[key] = val

            if "gen_pct" not in raw_row or "contiguous" not in raw_row:
                return None

            rec = {}

            # record_id
            v = raw_row.get("record_id", "").strip()
            rec["record_id"] = int(v) if v.isdigit() else None

            # gen_pct
            rec["gen_pct"] = float(str(raw_row["gen_pct"]).strip())

            # contiguous
            rec["contiguous"] = str(raw_row["contiguous"]).strip().lower() == "true"

            # ppl
            rec["ppl"] = _parse_ppl_cell(str(raw_row["ppl"])) if "ppl" in raw_row else None

            # se
            v = str(raw_row.get("se", "")).strip()
            rec["se"] = float(v) if v not in ("", None) else None

            # length
            v = str(raw_row.get("length", "")).strip()
            try:
                rec["length"] = int(v) if v != "" else None
            except Exception:
                rec["length"] = None

            # ptm
            v = str(raw_row.get("ptm", "")).strip()
            rec["ptm"] = float(v) if v not in ("", None) else None

            # mean_gen_plddt
            v = str(raw_row.get("mean_gen_plddt", "")).strip()
            rec["mean_gen_plddt"] = float(v) if v not in ("", None) else None

            # mean_non_gen_plddt
            v = str(raw_row.get("mean_non_gen_plddt", "")).strip()
            rec["mean_non_gen_plddt"] = float(v) if v not in ("", None) else None

            # idx (may be multiline)
            rec["idx"] = _parse_idx_cell(str(raw_row.get("idx", "")))

            # seq
            rec["seq"] = str(raw_row.get("seq", "")).strip()

            return rec

        for raw in f:
            line = raw.rstrip("\n")
            buf = (buf + "\n" + line) if buf else line

            if buf.count("\t") < (n_fields - 1):
                continue

            parts = buf.split("\t", n_fields - 1)
            buf = ""
            if len(parts) != n_fields:
                continue

            rec = _emit(parts)
            if rec is not None:
                rows.append(rec)

        # Flush if leftover buffer holds a complete record
        if buf and buf.count("\t") >= (n_fields - 1):
            parts = buf.split("\t", n_fields - 1)
            if len(parts) == n_fields:
                rec = _emit(parts)
                if rec is not None:
                    rows.append(rec)

    return rows

def parse_original_sequences_tsv(tsv_path):
    """
    Parse a TSV of original, non-generated protein sequences.

    Expected header (case-insensitive, order can vary):
        id, seq, ptm, mean_plddt

    Returns:
        List of dicts, each with keys:
            - id          (str or None)
            - seq         (str, possibly empty)
            - ptm         (float or None)
            - mean_plddt  (float or None)
    """
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return rows

        header = [h.strip().lower() for h in header_line.rstrip("\n").split("\t")]
        n_fields = len(header)

        def _idx(name):
            try:
                return header.index(name)
            except ValueError:
                return None

        idx_id      = _idx("id")
        idx_seq     = _idx("seq")
        idx_ptm     = _idx("ptm")
        idx_plddt   = _idx("mean_plddt")

        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            # Pad if line has fewer columns (trailing empty fields)
            if len(parts) < n_fields:
                parts += [""] * (n_fields - len(parts))

            rec = {}

            if idx_id is not None:
                rec["id"] = parts[idx_id].strip()
            else:
                rec["id"] = None

            if idx_seq is not None:
                rec["seq"] = parts[idx_seq].strip()
            else:
                rec["seq"] = ""

            # Helper to parse floats, treating "", "nan", "NaN" as None
            def _to_float(v):
                v = v.strip()
                if v == "" or v.lower() == "nan":
                    return None
                return float(v)

            rec["ptm"] = _to_float(parts[idx_ptm]) if idx_ptm is not None else None
            rec["mean_plddt"] = _to_float(parts[idx_plddt]) if idx_plddt is not None else None

            rows.append(rec)

    return rows
