# Dominance aware GWAS-pipeline

`dominance-gwas-pipeline` is a trimmed derivative of the original [GWAS-pipeline](https://github.com/sanchestm/GWAS-pipeline), reduced to the parts needed for dominance-aware SNP association testing.

This repo keeps:

- the edited `npplink.py` implementation with additive and additive+dominance GWAS
- a lightweight results summarizer for dominance hits
- simple Manhattan and locuszoom plotting helpers
- a minimal Conda environment

This repo does **not** keep the broader reporting, enrichment, database, phenotype-processing, or downstream QTL framework from the original project.

## What the model does

The dominance-aware model follows the encoding used in Cui et al. 2023, "Dominance is common in mammals and is associated with trans-acting gene expression and alternative splicing".

- Additive encoding: `0 / 1 / 2`
- Dominance encoding: `0 / 1 / 0`
- Joint model: additive + dominance
- Non-additive test: additive-only vs additive+dominance

Important result columns from the `add-dom` run:

- `neglog_p_additive`: additive-only association
- `neglog_p_add_joint`: additive term inside the joint additive+dominance model
- `neglog_p_dominance`: dominance term inside the joint model
- `neglog_p_avsad`: additive-only vs additive+dominance comparison
- `dominance_class`: heuristic classification (`A`, `PD`, `CD`, `OD`)

## Installation

```bash
git clone https://github.com/unculturedbacterium/dominance-gwas-pipeline.git
cd dominance-gwas-pipeline
conda env create -f environment.yml
conda activate dominance-gwas
```

## Included modules

- [dominance_gwas/npplink.py](dominance_gwas/npplink.py): PLINK reader, GRM utilities, additive GWAS, and additive+dominance GWAS
- [dominance_gwas/results.py](dominance_gwas/results.py): load and summarize `add-dom` parquet outputs
- [dominance_gwas/plotting.py](dominance_gwas/plotting.py): simple Manhattan and locuszoom plotting

## Minimal usage

```python
import pandas as pd
from dominance_gwas import GWAS

traits = pd.read_parquet("phenotypes.parquet")

GWAS(
    traitdf=traits[["trait_of_interest"]],
    genotypes="path/to/plink_prefix",
    grms_folder="path/to/grm_folder",
    save=True,
    save_path="results/gwas_addom/",
    return_table=False,
    y_correction="ystar",
    dtype="pandas_highmem",
    model="add-dom",
    regression_mode="einsum",
    center=True,
    scale=True,
)
```

This writes chromosome-wise files like:

- `gwas1.addom.parquet.gz`
- `gwas2.addom.parquet.gz`

## Summarizing add-dom results

```python
from dominance_gwas import summarize_adddom_results

summary = summarize_adddom_results(
    "results/gwas_addom/",
    threshold=7.5,
    save=True,
)
```

This writes:

- `dominance_summary.parquet.gz`
- `dominance_hits.csv`

## Plotting

```python
from dominance_gwas import load_adddom_results, manhattan_plot, locuszoom_plot

results = load_adddom_results("results/gwas_addom/")

manhattan_plot(
    results,
    score_col="neglog_p_dominance",
    threshold=7.5,
    output_path="manhattan_dominance.png",
)

locuszoom_plot(
    results,
    lead_snp="9:98405863",
    score_col="neglog_p_dominance",
    output_path="locuszoom_9_98405863.png",
)
```

## Notes

- This repo is focused on SNP-level dominance-aware GWAS only.
- The current plotting helpers are intentionally lightweight and do not reproduce the full original `core.py` plotting/report stack.
- The code is best suited for PLINK `bed/bim/fam` genotype inputs where SNP names follow the `CHR:POS` convention.

## Provenance

This repo was derived from the upstream `GWAS-pipeline` project and then reduced to a smaller dominance-focused codebase. The original project and license are preserved in spirit and attribution.
