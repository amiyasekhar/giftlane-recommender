# giftlane-recommender

DS 5020 Mini-Project "Customers Like You": a PCA/SVD recommender on UCI Online Retail II.

**Team.** Individual project: Amiya Sekhar (Data Engineer and ML Engineer).

## Success targets

Translated from Sarah's request (brief Section 1.4). Each is measured on data the model never saw.

1. **Smaller:** stored factors are much smaller than the dense matrix (dense A_log = 161.4 MB; target: factors < 5 MB at the chosen k).
2. **Fast:** the nightly job (fit + score all products) runs in under 60 seconds on a laptop.
3. **Better than best sellers:** hit rate @10 on held-out purchases beats the top-10 best-seller baseline; also report recall@10. Hit-rate ceiling = 99.07% (targets whose product is not in the matrix are unwinnable).
4. **Usable customer map:** 3-6 customer groups that marketing can name from their top products and countries, with no customer IDs shown.

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
`data/raw/online_retail_II.xlsx`. Unit tests: `python -m pytest tests`.

## Data

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). The raw file lives in `data/raw/` and is gitignored.

| Sheet | Rows |
| --- | ---: |
| Year 2009-2010 | 525,461 |
| Year 2010-2011 | 541,910 |
| **Total** | **1,067,371** |
