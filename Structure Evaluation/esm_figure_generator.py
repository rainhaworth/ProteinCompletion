# figure_generator.py
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def generate_figures_single(records, outdir="figs", plddt_scale_max=1.0):
    """
    records: iterable of dicts with keys:
      - gen_pct: float
      - contiguous: bool
      - ptm: float
      - mean_gen_plddt: float
      - mean_non_gen_plddt: float (not used here)
    outdir: directory to save all plots
    plddt_scale_max: 1.0 if pLDDT ∈ [0,1]; 100.0 if ∈ [0,100]
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records).rename(columns={"gen_pct": "generated"}).sort_values("generated")

    # ------------------------
    # LINE PLOTS
    # ------------------------

    # ==== pTM vs % generated (True & False on one plot, mean per % generated) ====
    df_ptm = df[["generated", "contiguous", "ptm"]].copy()

    plt.figure()
    sns.lineplot(
        data=df_ptm,
        x="generated", y="ptm",
        hue="contiguous",  # plot both True & False on same plot,
        hue_order=[True, False],  # Ensure legend order
        palette=["#440154", "#FDE725"],
        estimator="mean",  # average pTM for each % generated
        dashes=False,
        errorbar=None  # hide error bands
    )
    plt.title("pTM vs % Generated")
    plt.xlabel("% Generated")
    plt.ylabel("Mean pTM")
    plt.ylim([0, 1])
    plt.legend(title="Contiguous", labels=["True", "False"])
    plt.savefig(outdir / "ptm_vs_genpct_true_false.png", bbox_inches="tight")
    plt.close()

    # ==== Generated pLDDT vs % generated (True & False on one plot, mean per % generated) ====
    df_plddt = df[["generated", "contiguous", "mean_gen_plddt", "mean_non_gen_plddt"]].copy()

    plt.figure()
    sns.lineplot(
        data=df_plddt,
        x="generated", y="mean_gen_plddt",
        hue="contiguous",
        hue_order=[True, False],  # ensure legend order
        palette=["#440154", "#FDE725"],
        estimator="mean",
        dashes=False,
        errorbar=None
    )

    # Add straight line for global mean of non-generated pLDDT
    mean_non_gen = df_plddt["mean_non_gen_plddt"].mean()
    plt.axhline(
        y=mean_non_gen,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Mean Non-Generated pLDDT"
    )

    plt.title("Generated pLDDT vs % Generated")
    plt.xlabel("% Generated")
    plt.ylabel("Mean pLDDT")
    plt.ylim([0, plddt_scale_max])
    plt.legend(title="Contiguous / Non-Generated")
    plt.savefig(outdir / "plddt_vs_genpct_true_false.png", bbox_inches="tight")
    plt.close()

    # ------------------------
    # HISTOGRAMS
    # ------------------------

    # pTM histogram — contiguous=True
    sub_true = df[df["contiguous"] == True]
    if not sub_true.empty:
        plt.figure()
        sns.histplot(
            sub_true,
            x="ptm",
            bins=15,
            kde=False,
            color=sns.color_palette("viridis", 3)[1],
            edgecolor="white",
        )
        plt.title("pTM Distribution (contiguous=True)")
        plt.xlabel("pTM")
        plt.ylabel("Count")
        plt.xlim(0, 1)
        plt.savefig(outdir / "hist_ptm_contiguous_true.png", bbox_inches="tight")
        plt.close()

    # pTM histogram — contiguous=False
    sub_false = df[df["contiguous"] == False]
    if not sub_false.empty:
        plt.figure()
        sns.histplot(
            sub_false,
            x="ptm",
            bins=15,
            kde=False,
            color=sns.color_palette("magma", 3)[1],
            edgecolor="white",
        )
        plt.title("pTM Distribution (contiguous=False)")
        plt.xlabel("pTM")
        plt.ylabel("Count")
        plt.xlim(0, 1)
        plt.savefig(outdir / "hist_ptm_contiguous_false.png", bbox_inches="tight")
        plt.close()

    # Generated pLDDT histogram — contiguous=True
    if not sub_true.empty:
        plt.figure()
        sns.histplot(
            sub_true,
            x="mean_gen_plddt",
            bins=15,
            kde=False,
            color=sns.color_palette("viridis", 3)[2],
            edgecolor="white",
        )
        plt.title("Generated pLDDT Distribution (contiguous=True)")
        plt.xlabel("Mean Generated pLDDT")
        plt.ylabel("Count")
        plt.xlim(0, plddt_scale_max)
        plt.savefig(outdir / "hist_plddt_generated_contiguous_true.png", bbox_inches="tight")
        plt.close()

    # Generated pLDDT histogram — contiguous=False
    if not sub_false.empty:
        plt.figure()
        sns.histplot(
            sub_false,
            x="mean_gen_plddt",
            bins=15,
            kde=False,
            color=sns.color_palette("magma", 3)[2],
            edgecolor="white",
        )
        plt.title("Generated pLDDT Distribution (contiguous=False)")
        plt.xlabel("Mean Generated pLDDT")
        plt.ylabel("Count")
        plt.xlim(0, plddt_scale_max)
        plt.savefig(outdir / "hist_plddt_generated_contiguous_false.png", bbox_inches="tight")
        plt.close()

    # Non-Generated pLDDT histogram (global)
    plt.figure()
    sns.histplot(
        df,
        x="mean_non_gen_plddt",
        bins=15,
        kde=False,
        color="gray",
        edgecolor="white",
    )
    plt.title("Non-Generated pLDDT Distribution (Global)")
    plt.xlabel("Mean Non-Generated pLDDT")
    plt.ylabel("Count")
    plt.xlim(0, plddt_scale_max)
    plt.savefig(outdir / "hist_plddt_non_generated_global.png", bbox_inches="tight")
    plt.close()


def generate_figures_combined(bcm_records, esm_records, outdir, plddt_scale_max=1.0):
    """
    bcm_records, esm_records: iterables of dicts with keys:
      - gen_pct: float
      - contiguous: bool
      - ptm: float
      - mean_gen_plddt: float
      - mean_non_gen_plddt: float
    outdir: directory to save all plots
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Convert both record sets to DataFrames
    df_bcm = pd.DataFrame(bcm_records)
    df_esm = pd.DataFrame(esm_records)

    # Tag each dataset
    df_bcm["source"] = "BCM"
    df_esm["source"] = "ESM"

    # Combine into one DataFrame
    df = pd.concat([df_bcm, df_esm], ignore_index=True).sort_values("gen_pct")

    # Shared aesthetics
    source_palette = {"BCM": "#1f77b4", "ESM": "#d62728"}  # blue vs red
    contiguous_dashes = {True: "", False: (6, 3)}          # solid vs dashed

    # ------------------------
    # pTM vs % generated
    # ------------------------
    df_ptm = df[["gen_pct", "contiguous", "ptm", "source"]].copy()

    plt.figure(figsize=(7.5, 5))
    ax = sns.lineplot(
        data=df_ptm,
        x="gen_pct",
        y="ptm",
        hue="source",              # BCM vs ESM color
        style="contiguous",        # line style for contiguous
        palette=source_palette,
        dashes=contiguous_dashes,
        estimator="mean",
        errorbar=None,
        markers=False              # no dots or xs
    )
    ax.set_title("pTM vs % Generated (BCM and ESM)")
    ax.set_xlabel("% Generated")
    ax.set_ylabel("Mean pTM")
    ax.set_ylim(0, 1)

    # Legend (bottom, horizontal)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.22)
    ax.legend(
        title="Source / Contiguous",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(outdir / "ptm_vs_genpct_bcm_esm.png", bbox_inches="tight")
    plt.close()

    # ------------------------
    # pLDDT vs % generated (BCM & ESM) + non-generated baselines
    # ------------------------
    df_plddt = df[
        ["gen_pct", "contiguous", "mean_gen_plddt", "mean_non_gen_plddt", "source"]
    ].copy()

    plt.figure(figsize=(7.5, 5))
    ax = sns.lineplot(
        data=df_plddt,
        x="gen_pct",
        y="mean_gen_plddt",
        hue="source",
        style="contiguous",
        palette=source_palette,
        dashes=contiguous_dashes,
        estimator="mean",
        errorbar=None,
        markers=False
    )

    # Add per-source non-generated pLDDT baselines
    mean_non_gen_bcm = df_bcm["mean_non_gen_plddt"].mean()
    mean_non_gen_esm = df_esm["mean_non_gen_plddt"].mean()

    ax.axhline(
        y=mean_non_gen_bcm, color=source_palette["BCM"], linestyle="--", linewidth=1.5,
        label="Non-gen baseline (BCM)"
    )
    ax.axhline(
        y=mean_non_gen_esm, color=source_palette["ESM"], linestyle="--", linewidth=1.5,
        label="Non-gen baseline (ESM)"
    )

    ax.set_title("Generated pLDDT vs % Generated (BCM and ESM)")
    ax.set_xlabel("% Generated")
    ax.set_ylabel("Mean pLDDT")
    ax.set_ylim(0, plddt_scale_max)

    # Bottom legend, horizontal, with extra bottom space to avoid overlap
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.28)
    ax.legend(
        *zip(*uniq),
        title="Source / Contiguous / Non-Generated",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6
    )

    plt.tight_layout()
    plt.savefig(outdir / "plddt_vs_genpct_bcm_esm.png", bbox_inches="tight")
    plt.close()

    # ------------------------
    # HISTOGRAMS (BCM vs ESM overlaid)
    # ------------------------

    def hist_overlay(data, x, title, xlabel, xlim, outname):
        plt.figure()
        ax = sns.histplot(
            data=data,
            x=x,
            hue="source",
            multiple="layer",          # overlay
            bins=15,
            stat="count",
            common_bins=True,          # same bins across sources
            common_norm=False,
            kde=False,
            edgecolor="white",
            palette=source_palette,
            alpha=0.65,
            legend=False
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_xlim(xlim)

        # ---- Manual legend with color patches ----
        present = [s for s in data["source"].unique() if s in source_palette]
        handles = [Patch(facecolor=source_palette[s], edgecolor="white", alpha=0.65, label=s) for s in present]

        ax.legend(handles=handles, title="Source", loc="upper right", frameon=False)

        plt.tight_layout()
        plt.savefig(outdir / outname, bbox_inches="tight")
        plt.close()

    # Subsets by contiguous flag
    sub_true = df[df["contiguous"] == True]
    sub_false = df[df["contiguous"] == False]

    # 1) pTM histogram — contiguous=True (BCM & ESM)
    if not sub_true.empty:
        hist_overlay(
            sub_true,
            x="ptm",
            title="pTM Distribution (contiguous=True)",
            xlabel="pTM",
            xlim=(0, 1),
            outname="hist_ptm_contiguous_true.png",
        )

    # 2) pTM histogram — contiguous=False (BCM & ESM)
    if not sub_false.empty:
        hist_overlay(
            sub_false,
            x="ptm",
            title="pTM Distribution (contiguous=False)",
            xlabel="pTM",
            xlim=(0, 1),
            outname="hist_ptm_contiguous_false.png",
        )

    # 3) Generated pLDDT histogram — contiguous=True (BCM & ESM)
    if not sub_true.empty:
        hist_overlay(
            sub_true,
            x="mean_gen_plddt",
            title="Generated pLDDT Distribution (contiguous=True)",
            xlabel="Mean Generated pLDDT",
            xlim=(0, plddt_scale_max),
            outname="hist_plddt_generated_contiguous_true.png",
        )

    # 4) Generated pLDDT histogram — contiguous=False (BCM & ESM)
    if not sub_false.empty:
        hist_overlay(
            sub_false,
            x="mean_gen_plddt",
            title="Generated pLDDT Distribution (contiguous=False)",
            xlabel="Mean Generated pLDDT",
            xlim=(0, plddt_scale_max),
            outname="hist_plddt_generated_contiguous_false.png",
        )