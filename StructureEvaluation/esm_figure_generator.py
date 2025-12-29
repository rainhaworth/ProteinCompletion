# figure_generator.py
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def generate_figures_single(
    records,
    original_tsv,
    outdir,
    model_label="MODEL",
    model_color="blue",
    plddt_scale_max=1.0,
    length_ymax_plddt=None,
    length_ymax_ptm=None,
):
    """
    Single-model generator
    records: iterable of dicts with keys:
      - gen_pct: float/int
      - contiguous: bool
      - ptm: float
      - mean_gen_plddt: float
      - (optional) length: int   OR  seq: str  (needed for length plots)
      - (optional) mean_non_gen_plddt: float  (allowed)

    original_tsv: path to TSV of original, non-generated proteins with columns:
      - ptm
      - mean_plddt
      - (length plots need either) length OR seq

    Output files (key ones)
      - ptm_vs_genpct_{model}.png
      - plddt_vs_genpct_{model}.png
      - hist_ptm_contiguous_true_{model}.png
      - hist_ptm_contiguous_false_{model}.png
      - hist_plddt_generated_contiguous_true_{model}.png
      - hist_plddt_generated_contiguous_false_{model}.png
      - length_vs_plddt_all_genpct_combined_{model}.png
      - length_vs_plddt_facet_genpct_groups_{model}.png
      - length_vs_ptm_all_genpct_combined_{model}.png
      - length_vs_ptm_facet_genpct_groups_{model}.png
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_tag = str(model_label).lower()

    # ----------------------------
    # Load original baseline stats
    # ----------------------------
    df_orig = pd.read_csv(original_tsv, sep="\t")
    orig_mean_ptm = df_orig["ptm"].mean()
    orig_mean_plddt = df_orig["mean_plddt"].mean()

    # ----------------------------
    # Records DataFrame
    # ----------------------------
    df = pd.DataFrame(records).copy()
    if df.empty:
        return

    required = {"gen_pct", "contiguous", "ptm", "mean_gen_plddt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"records missing required keys: {sorted(missing)}")

    df = df.sort_values("gen_pct")

    # Style mappings
    style_map = {True: "-", False: "--"}  # contiguous solid, fragmented dashed
    row_order = [(False, "fragmented"), (True, "contiguous")]  # legend row order

    # ============================================================
    # Gen_pct line plots + histograms
    # ============================================================

    # ------------------------
    # pTM vs % generated
    # ------------------------
    df_ptm = df[["gen_pct", "ptm", "contiguous"]].copy()

    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()

    legend_handles = []
    for contig_flag, region_label in row_order:
        sub = df_ptm[df_ptm["contiguous"] == contig_flag]
        if sub.empty:
            continue

        sns.lineplot(
            data=sub,
            x="gen_pct",
            y="ptm",
            estimator="mean",
            errorbar=None,
            markers=False,
            color=model_color,
            linestyle=style_map[contig_flag],
            legend=False,
            ax=ax,
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=model_color,
                linestyle=style_map[contig_flag],
                label=f"{model_label} {region_label}",
            )
        )

    ax.axhline(y=orig_mean_ptm, linestyle="--", linewidth=1.5, color="black")

    ax.set_title(f"pTM vs % Generated ({model_label})")
    ax.set_xlabel("% Generated")
    ax.set_ylabel("Mean pTM")
    ax.set_ylim(0.1, 0.6)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    ax.legend(
        handles=legend_handles,
        title="Region",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(outdir / f"ptm_vs_genpct_{model_tag}.png", bbox_inches="tight")
    plt.close()

    # ------------------------
    # pLDDT vs % generated
    # ------------------------
    df_plddt = df[["gen_pct", "mean_gen_plddt", "contiguous"]].copy()

    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()

    legend_handles = []
    for contig_flag, region_label in row_order:
        sub = df_plddt[df_plddt["contiguous"] == contig_flag]
        if sub.empty:
            continue

        sns.lineplot(
            data=sub,
            x="gen_pct",
            y="mean_gen_plddt",
            estimator="mean",
            errorbar=None,
            markers=False,
            color=model_color,
            linestyle=style_map[contig_flag],
            legend=False,
            ax=ax,
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=model_color,
                linestyle=style_map[contig_flag],
                label=f"{model_label} {region_label}",
            )
        )

    ax.axhline(y=orig_mean_plddt, linestyle="--", linewidth=1.5, color="black")

    ax.set_title(f"Generated pLDDT vs % Generated ({model_label})")
    ax.set_xlabel("% Generated")
    ax.set_ylabel("Mean pLDDT")
    ax.set_ylim(0.4, min(plddt_scale_max, 0.8))  # same as combined

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.30)
    ax.legend(
        handles=legend_handles,
        title="Region",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(outdir / f"plddt_vs_genpct_{model_tag}.png", bbox_inches="tight")
    plt.close()

    # ------------------------
    # Histograms (split by contiguous)
    # ------------------------
    sub_true = df[df["contiguous"] == True]
    sub_false = df[df["contiguous"] == False]

    if not sub_true.empty:
        plt.figure()
        sns.histplot(sub_true, x="ptm", bins=15, kde=False, edgecolor="white", alpha=0.85)
        plt.title(f"pTM Distribution (contiguous=True) ({model_label})")
        plt.xlabel("pTM")
        plt.ylabel("Count")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(outdir / f"hist_ptm_contiguous_true_{model_tag}.png", bbox_inches="tight")
        plt.close()

    if not sub_false.empty:
        plt.figure()
        sns.histplot(sub_false, x="ptm", bins=15, kde=False, edgecolor="white", alpha=0.85)
        plt.title(f"pTM Distribution (contiguous=False) ({model_label})")
        plt.xlabel("pTM")
        plt.ylabel("Count")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(outdir / f"hist_ptm_contiguous_false_{model_tag}.png", bbox_inches="tight")
        plt.close()

    if not sub_true.empty:
        plt.figure()
        sns.histplot(sub_true, x="mean_gen_plddt", bins=15, kde=False, edgecolor="white", alpha=0.85)
        plt.title(f"Generated pLDDT Distribution (contiguous=True) ({model_label})")
        plt.xlabel("Mean Generated pLDDT")
        plt.ylabel("Count")
        plt.xlim(0, plddt_scale_max)
        plt.tight_layout()
        plt.savefig(outdir / f"hist_plddt_generated_contiguous_true_{model_tag}.png", bbox_inches="tight")
        plt.close()

    if not sub_false.empty:
        plt.figure()
        sns.histplot(sub_false, x="mean_gen_plddt", bins=15, kde=False, edgecolor="white", alpha=0.85)
        plt.title(f"Generated pLDDT Distribution (contiguous=False) ({model_label})")
        plt.xlabel("Mean Generated pLDDT")
        plt.ylabel("Count")
        plt.xlim(0, plddt_scale_max)
        plt.tight_layout()
        plt.savefig(outdir / f"hist_plddt_generated_contiguous_false_{model_tag}.png", bbox_inches="tight")
        plt.close()

    # ============================================================
    # (B) Plots based on sequence length
    # ============================================================

    def _ensure_length(df_in, name_for_errors):
        """Ensure df_in has a numeric 'length' column, inferred from 'seq' if needed."""
        df_out = df_in.copy()
        if "length" not in df_out.columns:
            if "seq" in df_out.columns:
                df_out["length"] = df_out["seq"].astype(str).str.len()
            else:
                raise ValueError(
                    f"{name_for_errors} must include 'length' or 'seq' to make length-based plots."
                )
        return df_out

    df_len_src = _ensure_length(df, "records")
    df_orig_len = _ensure_length(df_orig, "original_tsv")

    # Trim baseline to generated length range
    gen_len_min = df_len_src["length"].min()
    gen_len_max = df_len_src["length"].max()

    # Ensure gen_pct treated as categorical with stable ordering
    if not pd.api.types.is_categorical_dtype(df_len_src["gen_pct"]):
        unique_gen = sorted(df_len_src["gen_pct"].unique())
        df_len_src["gen_pct"] = pd.Categorical(df_len_src["gen_pct"], categories=unique_gen, ordered=True)

    def _length_plots(metric_kind, y_max):
        metric_kind = metric_kind.lower()
        if metric_kind == "plddt":
            y_col = "mean_gen_plddt"
            base_col = "mean_plddt"
            metric_label = "Mean pLDDT"
            fname_tag = "plddt"
        elif metric_kind == "ptm":
            y_col = "ptm"
            base_col = "ptm"
            metric_label = "pTM"
            fname_tag = "ptm"
        else:
            raise ValueError("metric_kind must be 'plddt' or 'ptm'")

        baseline_df = (
            df_orig_len
            .dropna(subset=[base_col, "length"])
            .groupby("length", as_index=False)[base_col]
            .mean()
            .sort_values("length")
        )

        baseline_df = baseline_df[
            (baseline_df["length"] >= gen_len_min) & (baseline_df["length"] <= gen_len_max)
        ]

        # -----------------------------------------
        # C) Combine all gen_pct values, avg per length
        # -----------------------------------------
        df_len = (
            df_len_src
            .dropna(subset=[y_col, "length"])
            .groupby(["length"], as_index=False)[y_col]
            .mean()
            .sort_values("length")
        )

        plt.figure(figsize=(6.5, 4.0))
        ax = sns.lineplot(
            data=df_len,
            x="length",
            y=y_col,
            estimator=None,
            errorbar=None,
            color=model_color,
        )

        ax.plot(
            baseline_df["length"],
            baseline_df[base_col],
            linestyle="--",
            linewidth=1.2,
            color="black",
            label=f"Original {metric_label}",
        )

        ax.set_title(f"{metric_label} vs Length (all % generated combined) ({model_label})")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(metric_label)
        if y_max is not None:
            ax.set_ylim(0, y_max)

        ax.legend(title="Baseline", frameon=False, loc="best")

        plt.tight_layout()
        plt.savefig(outdir / f"length_vs_{fname_tag}_all_genpct_combined_{model_tag}.png", bbox_inches="tight")
        plt.close()

        # -----------------------------------------
        # D) Group gen_pct into (20,40), (60,80), (90,95,99) and facet
        # -----------------------------------------
        df_group = df_len_src.copy()

        if pd.api.types.is_categorical_dtype(df_group["gen_pct"]):
            gen_numeric = df_group["gen_pct"].astype(float)
        else:
            gen_numeric = df_group["gen_pct"]

        def _genpct_group(val):
            if val in (20, 40):
                return "20–40%"
            elif val in (60, 80):
                return "60–80%"
            elif val in (90, 95, 99):
                return "90–99%"
            return None

        df_group["gen_pct_group"] = gen_numeric.map(_genpct_group)
        df_group = df_group.dropna(subset=["gen_pct_group", "length", y_col])

        if df_group.empty:
            return

        group_order = ["20–40%", "60–80%", "90–99%"]
        group_order = [g for g in group_order if g in df_group["gen_pct_group"].unique()]

        g_grp = sns.FacetGrid(
            df_group,
            col="gen_pct_group",
            col_order=group_order,
            height=3.0,
            sharex=True,
            sharey=True,
        )

        g_grp.map_dataframe(
            sns.lineplot,
            x="length",
            y=y_col,
            estimator="mean",
            errorbar="sd",
            color=model_color,
        )

        # Add baseline to each facet
        for ax in g_grp.axes.flatten():
            if ax is not None:
                ax.plot(
                    baseline_df["length"],
                    baseline_df[base_col],
                    linestyle="--",
                    linewidth=1.2,
                    color="black",
                    label=f"Original {metric_label}",
                )

        g_grp.set_axis_labels("Sequence Length", metric_label)
        g_grp.set_titles(col_template="gen % group = {col_name}", pad=22)
        if y_max is not None:
            g_grp.set(ylim=(0, y_max))

        # One legend (baseline only) — place nicely
        g_grp.add_legend(title="Baseline")
        leg = g_grp._legend
        leg.set_bbox_to_anchor((0.95, 0.65))
        leg.set_loc("center")

        g_grp.fig.suptitle(f"{metric_label} vs Length by % Generated Groups ({model_label})", y=1.04)
        g_grp.fig.tight_layout()
        g_grp.fig.savefig(outdir / f"length_vs_{fname_tag}_facet_genpct_groups_{model_tag}.png", bbox_inches="tight")
        plt.close(g_grp.fig)

    # Make both metric kinds
    _length_plots("plddt", y_max=length_ymax_plddt)
    _length_plots("ptm", y_max=length_ymax_ptm)

def generate_figures_combined(bcm_records, esm_records, original_tsv, outdir, plddt_scale_max=1.0):
    """
    bcm_records, esm_records: iterables of dicts with keys:
      - gen_pct: float
      - contiguous: bool
      - ptm: float
      - mean_gen_plddt: float
      - mean_non_gen_plddt: float
    original_tsv: path to TSV with original, non-generated proteins with columns:
      - ptm
      - mean_plddt
    outdir: directory to save all plots
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load original, non-generated baseline statistics
    df_orig = pd.read_csv(original_tsv, sep="\t")
    orig_mean_ptm = df_orig["ptm"].mean()
    orig_mean_plddt = df_orig["mean_plddt"].mean()

    # Convert both record sets to DataFrames
    df_bcm = pd.DataFrame(bcm_records)
    df_esm = pd.DataFrame(esm_records)

    # Tag each dataset
    df_bcm["source"] = "BCM"
    df_esm["source"] = "ESM"

    # Combine into one DataFrame
    df = pd.concat([df_bcm, df_esm], ignore_index=True).sort_values("gen_pct")

    # (Optionally keep this; no longer used for plotting)
    df["line_id"] = df.apply(
        lambda r: f"{r['source']} - {'contiguous' if r['contiguous'] else 'fragmented'}",
        axis=1
    )

    # Common style mappings: ESM red, BCM blue; contiguous solid, fragmented dashed
    color_map = {"ESM": "red", "BCM": "blue"}
    style_map = {True: "-", False: "--"}
    row_order = [(False, "fragmented"), (True, "contiguous")]  # legend rows
    col_order = [("ESM", "ESM"), ("BCM", "BCM")]  # legend columns

    # ------------------------
    # pTM vs % generated (each line: model + contiguous/fragmented)
    # ------------------------
    df_ptm = df[["gen_pct", "ptm", "source", "contiguous"]].copy()

    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()

    legend_handles = []

    # Plot lines in row-major order for legend layout:
    # row 1: fragmented (ESM, BCM)
    # row 2: contiguous (ESM, BCM)
    for contig_flag, region_label in row_order:
        for source_key, source_label in col_order:
            sub = df_ptm[(df_ptm["source"] == source_key) &
                         (df_ptm["contiguous"] == contig_flag)]
            if sub.empty:
                continue

            sns.lineplot(
                data=sub,
                x="gen_pct",
                y="ptm",
                estimator="mean",
                errorbar=None,
                markers=False,
                color=color_map[source_key],
                linestyle=style_map[contig_flag],
                legend=False,
                ax=ax,
            )

            # One legend handle per (model, region) combo
            handle = Line2D(
                [0], [0],
                color=color_map[source_key],
                linestyle=style_map[contig_flag],
                label=f"{source_label} {region_label}",
            )
            legend_handles.append(handle)

    # Baseline from original, non-generated proteins (no legend label)
    ax.axhline(
        y=orig_mean_ptm,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    ax.set_title("pTM vs % Generated (BCM and ESM)")
    ax.set_xlabel("% Generated")
    ax.set_ylabel("Mean pTM")
    ax.set_ylim(0.1, 0.6)  # tighter y-range as requested

    # Custom legend: 2 columns (ESM, BCM), rows = fragmented/contiguous
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    ax.legend(
        handles=legend_handles,
        title="Model / Region",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(outdir / "ptm_vs_genpct_bcm_esm.png", bbox_inches="tight")
    plt.close()

    # ------------------------
    # pLDDT vs % generated (each line: model + contiguous/fragmented) + global baseline
    # ------------------------
    df_plddt = df[["gen_pct", "mean_gen_plddt", "source", "contiguous"]].copy()

    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()

    legend_handles = []

    for contig_flag, region_label in row_order:
        for source_key, source_label in col_order:
            sub = df_plddt[(df_plddt["source"] == source_key) &
                           (df_plddt["contiguous"] == contig_flag)]
            if sub.empty:
                continue

            sns.lineplot(
                data=sub,
                x="gen_pct",
                y="mean_gen_plddt",
                estimator="mean",
                errorbar=None,
                markers=False,
                color=color_map[source_key],
                linestyle=style_map[contig_flag],
                legend=False,
                ax=ax,
            )

            handle = Line2D(
                [0], [0],
                color=color_map[source_key],
                linestyle=style_map[contig_flag],
                label=f"{source_label} {region_label}",
            )
            legend_handles.append(handle)

    # Baseline from original, non-generated proteins (no legend label)
    ax.axhline(
        y=orig_mean_plddt,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    ax.set_title("Generated pLDDT vs % Generated (BCM and ESM)")
    ax.set_xlabel("% Generated")
    ax.set_ylabel("Mean pLDDT")
    ax.set_ylim(0.4, min(plddt_scale_max, 0.8))  # tighter range as suggested

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.30)
    ax.legend(
        handles=legend_handles,
        title="Model / Region",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    plt.tight_layout()
    plt.savefig(outdir / "plddt_vs_genpct_bcm_esm.png", bbox_inches="tight")
    plt.close()

    # ------------------------
    # HISTOGRAMS (BCM vs ESM stacked instead of overlaid)
    # ------------------------

    # Subsets by contiguous flag
    sub_true = df[df["contiguous"] == True]
    sub_false = df[df["contiguous"] == False]

    def hist_stack(data, x, title, xlabel, xlim, outname):
        plt.figure()

        # Decide the order once
        hue_order = list(data["source"].unique())

        ax = sns.histplot(
            data=data,
            x=x,
            hue="source",
            hue_order=hue_order,  # enforce same order
            multiple="stack",
            bins=15,
            stat="count",
            common_bins=True,
            common_norm=False,
            kde=False,
            edgecolor="white",
            alpha=0.85,
            legend=False
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_xlim(xlim)

        colors = sns.color_palette(n_colors=len(hue_order))

        handles = []
        for color, src in zip(colors, hue_order):
            patch = Patch(
                facecolor=color,
                edgecolor="white",
                alpha=0.85,
                label=src
            )
            handles.append(patch)

        ax.legend(handles=handles, title="Source", loc="upper right", frameon=False)

        plt.tight_layout()
        plt.savefig(outdir / outname, bbox_inches="tight")
        plt.close()

    # 1) pTM histogram — contiguous=True (BCM & ESM)
    if not sub_true.empty:
        hist_stack(
            sub_true,
            x="ptm",
            title="pTM Distribution (contiguous=True)",
            xlabel="pTM",
            xlim=(0, 1),
            outname="hist_ptm_contiguous_true.png",
        )

    # 2) pTM histogram — contiguous=False (BCM & ESM)
    if not sub_false.empty:
        hist_stack(
            sub_false,
            x="ptm",
            title="pTM Distribution (contiguous=False)",
            xlabel="pTM",
            xlim=(0, 1),
            outname="hist_ptm_contiguous_false.png",
        )

    # 3) Generated pLDDT histogram — contiguous=True (BCM & ESM)
    if not sub_true.empty:
        hist_stack(
            sub_true,
            x="mean_gen_plddt",
            title="Generated pLDDT Distribution (contiguous=True)",
            xlabel="Mean Generated pLDDT",
            xlim=(0, plddt_scale_max),
            outname="hist_plddt_generated_contiguous_true.png",
        )

    # 4) Generated pLDDT histogram — contiguous=False (BCM & ESM)
    if not sub_false.empty:
        hist_stack(
            sub_false,
            x="mean_gen_plddt",
            title="Generated pLDDT Distribution (contiguous=False)",
            xlabel="Mean Generated pLDDT",
            xlim=(0, plddt_scale_max),
            outname="hist_plddt_generated_contiguous_false.png",
        )

def plot_length_genpct_metric_all(
    bcm_records,
    esm_records,
    orig_records,
    metric_kind,
    outdir,
    y_max=None,
):
    """
    Make Matplotlib + Seaborn + Plotly plots for:
        sequence length vs (pLDDT or pTM), conditioned on gen_pct.

    Baseline is now a function of length: mean metric over original,
    non-generated proteins as a function of sequence length.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Build DataFrames from records
    # ----------------------------
    df_bcm = pd.DataFrame(bcm_records)
    df_esm = pd.DataFrame(esm_records)
    df_orig = pd.DataFrame(orig_records)

    # Tag sources
    df_bcm["source"] = "BCM"
    df_esm["source"] = "ESM"

    # Combined DataFrame for generated-completion experiments
    df = pd.concat([df_bcm, df_esm], ignore_index=True)

    # ----------------------------
    # Decide which columns / labels to use
    # ----------------------------
    metric_kind = metric_kind.lower()
    if metric_kind == "plddt":
        y_col = "mean_gen_plddt"   # from generated completions
        base_col = "mean_plddt"    # from original, non-generated sequences
        metric_label = "Mean pLDDT"
        fname_tag = "plddt"
    elif metric_kind == "ptm":
        y_col = "ptm"
        base_col = "ptm"
        metric_label = "pTM"
        fname_tag = "ptm"
    else:
        raise ValueError("metric_kind must be 'plddt' or 'ptm'")

    # ----------------------------
    # Build length-dependent baseline from original sequences
    # ----------------------------

    # Ensure original has length; infer from seq if needed
    if "length" not in df_orig.columns:
        if "seq" in df_orig.columns:
            df_orig["length"] = df_orig["seq"].astype(str).str.len()
        else:
            raise ValueError("orig_records must include 'length' or 'seq' to compute a baseline curve.")

    baseline_df = (
        df_orig
        .dropna(subset=[base_col, "length"])
        .groupby("length", as_index=False)[base_col]
        .mean()
        .sort_values("length")
    )

    # # (Optional) smooth baseline a bit
    # if len(baseline_df) > 5:
    #     baseline_df["baseline_smooth"] = (
    #         baseline_df[base_col]
    #         .rolling(window=25, min_periods=3, center=True)
    #         .mean()
    #         .bfill()
    #         .ffill()
    #     )
    #     baseline_y = "baseline_smooth"
    # else:
    #     baseline_y = base_col

    baseline_y = base_col

    # Trim baseline to generated length range
    # Ensure generated df has length; infer from seq if needed
    if "length" not in df.columns:
        if "seq" in df.columns:
            df["length"] = df["seq"].astype(str).str.len()
        else:
            raise ValueError(
                "bcm/esm records must include 'length' or 'seq' to plot vs length."
            )

    gen_len_min = df["length"].min()
    gen_len_max = df["length"].max()

    # This will typically correspond to something like [119, 948]
    baseline_df = baseline_df[
        (baseline_df["length"] >= gen_len_min)
        & (baseline_df["length"] <= gen_len_max)
    ]

    # Ensure gen_pct is treated as categorical with a stable ordering
    if not pd.api.types.is_categorical_dtype(df["gen_pct"]):
        unique_gen = sorted(df["gen_pct"].unique())
        df["gen_pct"] = pd.Categorical(df["gen_pct"], categories=unique_gen, ordered=True)

    # ---------------------------------------------------------------------
    # C) Idea 1: Combine all gen_pct values, average per length (per source)
    # ---------------------------------------------------------------------
    df_len = (
        df
        .dropna(subset=[y_col, "length"])
        .groupby(["source", "length"], as_index=False)[y_col]
        .mean()
        .sort_values("length")
    )

    plt.figure(figsize=(6.5, 4.0))
    ax = sns.lineplot(
        data=df_len,
        x="length",
        y=y_col,
        hue="source",
        estimator=None,
        errorbar=None,
    )

    # Add baseline (length-dependent)
    ax.plot(
        baseline_df["length"],
        baseline_df[baseline_y],
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=f"Original {metric_label}",
    )

    ax.set_title(f"{metric_label} vs Length (all % generated combined)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(metric_label)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    ax.legend(title="Source / Baseline", frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(outdir / f"length_vs_{fname_tag}_all_genpct_combined.png", bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------------
    # D) Idea 2: Group gen_pct into (20,40), (60,80), (90,95,99) and facet
    # ---------------------------------------------------------------------
    df_group = df.copy()

    # Work with numeric gen_pct values
    if pd.api.types.is_categorical_dtype(df_group["gen_pct"]):
        gen_numeric = df_group["gen_pct"].astype(float)
    else:
        gen_numeric = df_group["gen_pct"]

    def _genpct_group(val):
        if val in (20, 40):
            return "20–40%"
        elif val in (60, 80):
            return "60–80%"
        elif val in (90, 95, 99):
            return "90–99%"
        else:
            return None

    df_group["gen_pct_group"] = gen_numeric.map(_genpct_group)
    df_group = df_group.dropna(subset=["gen_pct_group", "length", y_col])

    if not df_group.empty:
        group_order = ["20–40%", "60–80%", "90–99%"]
        # Filter order to only those groups actually present
        group_order = [g for g in group_order if g in df_group["gen_pct_group"].unique()]

        g_grp = sns.FacetGrid(
            df_group,
            col="gen_pct_group",
            col_order=group_order,
            hue="source",
            height=3.0,
            sharex=True,
            sharey=True,
        )

        g_grp.map_dataframe(
            sns.lineplot,
            x="length",
            y=y_col,
            estimator="mean",
            errorbar="sd",
        )

        # Add baseline to each facet
        first_baseline = True
        for ax in g_grp.axes.flatten():
            if ax is not None:
                label = f"Original {metric_label}" if first_baseline else "_nolegend_"
                ax.plot(
                    baseline_df["length"],
                    baseline_df[baseline_y],
                    linestyle="--",
                    linewidth=1.2,
                    color="black",
                    label=label,
                )
                first_baseline = False

        g_grp.set_axis_labels("Sequence Length", metric_label)
        g_grp.set_titles(col_template="gen % group = {col_name}", pad=22)
        if y_max is not None:
            g_grp.set(ylim=(0, y_max))

        g_grp.add_legend(title="Source / Baseline")

        # Move legend upward
        leg = g_grp._legend
        leg.set_bbox_to_anchor((0.95, 0.65))  # shift vertically above the facets
        leg.set_loc("center")  # center horizontally

        # Keep them visually "in one row" by not wrapping (since max 3 groups)
        g_grp.fig.suptitle(
            f"{metric_label} vs Length by % Generated Groups", y=1.04
        )
        g_grp.fig.tight_layout()
        g_grp.fig.savefig(
            outdir / f"length_vs_{fname_tag}_facet_genpct_groups.png",
            bbox_inches="tight",
        )
        plt.close(g_grp.fig)