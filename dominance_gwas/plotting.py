from __future__ import annotations

import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "dominance_gwas_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_chr_pos(df: pd.DataFrame, snp_col: str = "snp") -> pd.DataFrame:
    out = df.copy()
    if "chrom" not in out.columns or "pos" not in out.columns:
        parsed = out[snp_col].astype(str).str.split(":", n=1, expand=True)
        out["chrom"] = parsed[0]
        out["pos"] = parsed[1].astype(int)
    return out


def manhattan_plot(
    results: pd.DataFrame,
    score_col: str = "neglog_p_additive",
    snp_col: str = "snp",
    threshold: float | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 5),
):
    df = _ensure_chr_pos(results, snp_col=snp_col).copy()
    df = df.dropna(subset=[score_col]).sort_values(["chrom", "pos"]).reset_index(drop=True)
    chrom_order = sorted(df["chrom"].astype(str).unique(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    colors = ["#203864", "#7f6000"]

    x_ticks = []
    x_labels = []
    offset = 0
    xs = np.zeros(len(df), dtype=float)
    for chrom_idx, chrom in enumerate(chrom_order):
        mask = df["chrom"].astype(str) == chrom
        pos = df.loc[mask, "pos"].to_numpy()
        shifted = pos + offset
        xs[mask.to_numpy()] = shifted
        x_ticks.append((shifted.min() + shifted.max()) / 2)
        x_labels.append(chrom)
        offset = shifted.max() + 1_000_000

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    for chrom_idx, chrom in enumerate(chrom_order):
        mask = df["chrom"].astype(str) == chrom
        ax.scatter(xs[mask.to_numpy()], df.loc[mask, score_col], s=8, color=colors[chrom_idx % len(colors)], linewidths=0)
    if threshold is not None:
        ax.axhline(threshold, color="#b22222", linestyle="--", linewidth=1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(f"-log10(P) [{score_col}]")
    ax.set_title(title or "Manhattan Plot")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def locuszoom_plot(
    results: pd.DataFrame,
    lead_snp: str,
    score_col: str = "neglog_p_additive",
    snp_col: str = "snp",
    window: int = 500_000,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
):
    df = _ensure_chr_pos(results, snp_col=snp_col).copy()
    lead = df.loc[df[snp_col].astype(str) == str(lead_snp)]
    if lead.empty:
        raise ValueError(f"{lead_snp} not found in results")
    lead = lead.iloc[0]
    chrom = str(lead["chrom"])
    pos = int(lead["pos"])
    region = df[
        (df["chrom"].astype(str) == chrom)
        & (df["pos"] >= pos - window)
        & (df["pos"] <= pos + window)
    ].dropna(subset=[score_col]).sort_values("pos")
    if region.empty:
        raise ValueError("No variants found in the requested locus window")

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.scatter(region["pos"], region[score_col], s=16, color="#4c78a8", alpha=0.8, linewidths=0)
    ax.scatter([pos], [lead[score_col]], s=36, color="#d62728", zorder=3)
    ax.axvline(pos, color="#d62728", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Chr {chrom} position (bp)")
    ax.set_ylabel(f"-log10(P) [{score_col}]")
    ax.set_title(f"Locuszoom: {lead_snp}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax, region
