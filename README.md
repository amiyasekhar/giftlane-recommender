# giftlane-recommender

DS 5020 Mini-Project "Customers Like You": a PCA/SVD recommender on UCI Online Retail II.

**Team.** Individual project: Amiya Sekhar (Data Engineer and ML Engineer).

## Summary

This project turns two years of GiftLane purchase history into a Truncated SVD model with 50 "taste patterns" that
powers a product-page "customers who bought this also bought" box, personalised top-10 lists, and five named customer
groups for marketing. Tested on the final two months of purchases the model never saw, a plain best-seller list still
wins on hit rate (26.8% vs 21.4% of customers with at least one correct suggestion), while the model catches more of
each customer's new purchases (recall 2.35% vs 1.74%), especially for light buyers, and gives personal lists with
almost no overlap with the best sellers. The recommendation is to use the model's "also bought" table on product pages
now, show best sellers to heavy buyers and model suggestions to light buyers, and test a binary purchase value plus a
popularity blend on a fresh time period before any further switch.

## Success targets

Translated from Sarah's request (brief Section 1.4). Each is measured on data the model never saw.

1. **Smaller:** stored factors are much smaller than the dense matrix (dense A_log = 161.4 MB; target: factors < 5 MB at the chosen k).
2. **Fast:** the nightly job (fit + score all products) runs in under 60 seconds on a laptop.
3. **Better than best sellers:** hit rate @10 on held-out purchases beats the top-10 best-seller baseline; also report recall@10. Hit-rate ceiling = 99.07% (targets whose product is not in the matrix are unwinnable).
4. **Usable customer map:** 3-6 customer groups that marketing can name from their top products and countries, with no customer IDs shown.

## Results against targets

| Target | Result | Met? |
| --- | --- | --- |
| Smaller | Factors 3.66 MB at k = 50 vs 161.4 MB dense (`reports/ml2_k_results.csv`, `reports/matrix_report.csv`) | Yes |
| Fast | Whole nightly job (fit, item neighbours, score all 5,471 customers) about 0.4 s (k = 50, median of 3) vs 60 s target (notebook 07) | Yes |
| Better than best sellers | hit@10 21.4% vs baseline 26.8% (not met); recall@10 2.35% vs 1.74% (better); light buyers with 1-10 products: recall 8.3% vs 2.4% (`reports/ml4_results.csv`, `reports/ml4_breakdowns.csv`) | No (hit@10), explained in `reports/memo.pdf` |
| Usable map | 5 named segments, sizes 29.7 / 23.9 / 20.0 / 13.3 / 13.2%, no customer IDs (`reports/ml6_segments_k5.csv`, `reports/figures/ml6_map_k5.png`) | Yes |

## Setup (from a fresh clone)

```bash
conda create -n giftlane python=3.11 -y
conda activate giftlane
python -m pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/online_retail_ii.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip data/raw/online_retail_ii.zip -d data/raw/
```

Expected: `data/raw/online_retail_II.xlsx`, about 44 MB. Everything under `data/` is gitignored.

Every notebook and `build_matrix.py` start with an environment guard (`src/env_check.py`) that requires Python 3.11
and pandas 3.0.6 and otherwise stops with "Wrong Python environment. Run: conda activate giftlane". Set
`GIFTLANE_SKIP_ENV_CHECK=1` to bypass it.

## How to run

```bash
conda activate giftlane   # or call the env's python directly
python -m pip install -r requirements.txt
python build_matrix.py
```

`build_matrix.py` runs the whole data pipeline from the raw file in one command: clean (`src/clean.py`) → time-based
split (`src/split.py`) → sparse customer × product matrices (`src/build_matrix.py`; log, binary, raw) → test-target
flagging. It writes every artefact to `data/processed/` (gitignored) and the reports to `reports/`, exactly as
notebooks `02_clean`, `03_split` and `04_matrix` do. The raw workbook must already be at
`data/raw/online_retail_II.xlsx`. It takes about 26 s on a laptop (reading the Excel file is most of it) and its
output is byte-identical to the notebooks' output. Unit tests: `python -m pytest tests`.

## Notebooks (run order)

Select the **Python (giftlane)** kernel. Run `build_matrix.py` (or notebooks 01-04) before 05-10, which read the
frozen matrices from `data/processed/`.

| Notebook | Step | One line |
| --- | --- | --- |
| `01_profile.ipynb` | DE1 | Profile the raw workbook: sheets, types, missing values, quantity and price ranges |
| `02_clean.ipynb` | DE2 | Cleaning rules in fixed order with a per-rule log (`reports/cleaning_log.csv`) |
| `03_split.ipynb` | DE3 | Time-based split, last 2 months held out, leakage checks, test targets |
| `04_matrix.ipynb` | DE4/DE5 | Sparse customer × product matrices (log, binary, raw), product filter, matrix report |
| `05_svd_k.ipynb` | ML2 | Truncated SVD k sweep, error formula validated by hand, scree and error curves |
| `06_recommend.ipynb` | ML3 | Product vectors, item neighbours ("also bought"), customer top-10 lists, best-seller baseline |
| `07_evaluate.ipynb` | ML4 | Evaluation on held-out purchases: 12 configs vs baseline, bootstrap, breakdowns, timing |
| `08_pca_vs_svd.ipynb` | ML5 | PCA vs Truncated SVD on the same sparse matrix, brief comparison table |
| `09_customer_map.ipynb` | ML6 | Unit-norm customer vectors, k-means for 3-6 groups, profiles, 2-D maps, named 5-segment solution |
| `10_sensitivity.ipynb` | ML7 | Sensitivity to the purchase value: raw vs binary vs log |

## Repository map

```
build_matrix.py        one-command data pipeline (clean -> split -> matrices), seed 42
decision_log.md        every method decision with reason and evidence, in order
requirements.txt       pinned packages for the giftlane env (pandas 3.0.6, scikit-learn, python-pptx, xhtml2pdf, ...)
src/
  env_check.py         environment guard (Python 3.11 + pandas 3.0.6)
  clean.py             cleaning rules and log
  split.py             time-based split and leakage checks
  build_matrix.py      sparse matrices, product filter, test-target flags
  svd.py               Truncated SVD fit, energy / error helpers, component inspection
  recommend.py         product vectors, item neighbours, customer scoring, best-seller baseline
  evaluate.py          evaluation set, hit/recall/precision, paired bootstrap, breakdowns
  segments.py          customer vectors, k-means sweep, segment profiles, 2-D maps
notebooks/             01-10 as listed above
tests/                 pytest: toy-matrix build (incl. the log-aggregation bug) and recommender checks
reports/
  ml1_decision.md      why Truncated SVD was chosen before fitting
  ml5_comparison.md    PCA vs SVD brief table and why they agree here
  memo.md / memo.pdf   one-page memo to the Head of E-Commerce
  slides.pptx          5-minute presentation with speaker notes and backup slides
  *.csv                cleaning log, data profile, matrix report, k sweep, ML4/ML5/ML6/ML7 results
  figures/             k-sweep curves, ML4 results, ML6 customer maps (k = 3-6), ML7 sensitivity
```

Main documents: `reports/ml1_decision.md`, `reports/ml5_comparison.md`, `reports/memo.pdf`, `reports/slides.pptx`
and `decision_log.md`.

## Data

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). The raw file lives in `data/raw/` and is gitignored.

| Sheet | Rows |
| --- | ---: |
| Year 2009-2010 | 525,461 |
| Year 2010-2011 | 541,910 |
| **Total** | **1,067,371** |

