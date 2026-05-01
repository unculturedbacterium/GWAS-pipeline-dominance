from __future__ import annotations

import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "dominance_gwas_mplconfig")

import holoviews as hv
import hvplot.pandas  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

hv.extension("bokeh")


def _ensure_chr_pos(df: pd.DataFrame, snp_col: str = "snp") -> pd.DataFrame:
    out = df.copy()
    if "chrom" not in out.columns or "pos" not in out.columns:
        parsed = out[snp_col].astype(str).str.split(":", n=1, expand=True)
        out["chrom"] = parsed[0]
        out["pos"] = parsed[1].astype(int)
    return out


def _chrom_sort_key(chrom: str):
    chrom = str(chrom)
    return (not chrom.isdigit(), int(chrom) if chrom.isdigit() else chrom)


def _prepare_manhattan_frame(results: pd.DataFrame, score_col: str, snp_col: str) -> tuple[pd.DataFrame, list[float], list[str]]:
    df = _ensure_chr_pos(results, snp_col=snp_col).copy()
    df = df.dropna(subset=[score_col]).sort_values(["chrom", "pos"]).reset_index(drop=True)
    chrom_order = sorted(df["chrom"].astype(str).unique(), key=_chrom_sort_key)
    x_ticks: list[float] = []
    x_labels: list[str] = []
    offset = 0
    xs = np.zeros(len(df), dtype=np.int64)
    color_group = np.zeros(len(df), dtype=np.int8)
    for chrom_idx, chrom in enumerate(chrom_order):
        mask = df["chrom"].astype(str) == chrom
        pos = df.loc[mask, "pos"].to_numpy(dtype=np.int64)
        shifted = pos + offset
        xs[mask.to_numpy()] = shifted
        color_group[mask.to_numpy()] = chrom_idx % 2
        x_ticks.append(float((shifted.min() + shifted.max()) / 2))
        x_labels.append(chrom)
        offset = int(shifted.max() + 1_000_000)
    df["genome_pos"] = xs
    df["chrom_group"] = np.where(color_group == 0, "even_chr", "odd_chr")
    return df, x_ticks, x_labels


def manhattan_plot(
    results: pd.DataFrame,
    score_col: str = "neglog_p_additive",
    snp_col: str = "snp",
    threshold: float | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 5),
):
    df, x_ticks, x_labels = _prepare_manhattan_frame(results, score_col=score_col, snp_col=snp_col)
    width = int(figsize[0] * 120)
    height = int(figsize[1] * 120)

    plot = df.hvplot.scatter(
        x="genome_pos",
        y=score_col,
        by="chrom_group",
        datashade=True,
        dynspread=True,
        aggregator="max",
        cmap=["#203864", "#7f6000"],
        xlabel="Chromosome",
        ylabel=f"-log10(P) [{score_col}]",
        title=title or "Manhattan Plot",
        width=width,
        height=height,
        legend=False,
        tools=["hover"],
        hover_cols=[snp_col, "chrom", "pos", score_col],
    ).opts(
        xrotation=0,
        xticks=list(zip(x_ticks, x_labels)),
        fontsize={"title": 14, "labels": 12, "xticks": 10, "yticks": 10},
    )

    if threshold is not None:
        plot = plot * hv.HLine(threshold).opts(color="#b22222", line_dash="dashed", line_width=1.5)

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix.lower() not in {".html", ".htm"}:
            output_path = output_path.with_suffix(".html")
        hv.save(plot, str(output_path), backend="bokeh")
    return plot


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
