from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_adddom_results(results_path: str | Path) -> pd.DataFrame:
    results_path = Path(results_path)
    files = sorted(results_path.glob("*.addom.parquet.gz"))
    if not files:
        raise FileNotFoundError(f"No .addom.parquet.gz files found in {results_path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def summarize_adddom_results(
    results_path: str | Path,
    threshold: float,
    save: bool = True,
) -> pd.DataFrame:
    out = load_adddom_results(results_path)
    out["is_sig_additive"] = out["neglog_p_additive"] >= threshold
    out["is_sig_dominance"] = out["neglog_p_dominance"] >= threshold
    out["is_sig_nonadditive"] = out["neglog_p_avsad"] >= threshold
    out["effect_pattern"] = np.select(
        [
            out["is_sig_additive"] & out["is_sig_nonadditive"],
            out["is_sig_nonadditive"],
            out["is_sig_additive"],
        ],
        [
            "additive+dominance",
            "dominance",
            "additive",
        ],
        default="ns",
    )
    if save:
        results_path = Path(results_path)
        out.to_parquet(results_path / "dominance_summary.parquet.gz", compression="gzip", index=False)
        out[out["effect_pattern"] != "ns"].to_csv(results_path / "dominance_hits.csv", index=False)
    return out
