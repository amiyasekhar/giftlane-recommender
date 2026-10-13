#!/usr/bin/env python
"""DE6 - run the whole data pipeline from the raw workbook in one command.

    python build_matrix.py

Stages (same code the notebooks call, same outputs):

    1. src/clean.py        data/raw/online_retail_II.xlsx -> data/processed/clean.parquet,
                                                            reports/cleaning_log.csv
    2. src/split.py        clean.parquet -> data/processed/train.parquet,
                                            data/processed/test_targets.parquet (seen_in_train)
    3. src/build_matrix.py train.parquet -> data/processed/A_log.npz, A_binary.npz, A_raw.npz,
                                            customer_index.parquet, product_index.parquet,
                                            reports/matrix_report.csv
    4. flag targets        test_targets.parquet overwritten with the in_matrix column

Nothing in data/raw/ is modified. Every stage prints one summary line; the matrix shape,
nnz and sparsity are printed at the end.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import build_matrix as bm  # noqa: E402
from src import clean as cl  # noqa: E402
from src import split as sp_  # noqa: E402

SEED = 42
RAW_PATH = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MIN_CUSTOMERS = 10
VALUE = "log"


def _stage(n: int, name: str, t0: float, msg: str) -> None:
    print(f"[{n}] {name:<14} {time.perf_counter() - t0:6.1f}s | {msg}", flush=True)


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)

    if not RAW_PATH.exists():
        print(f"ERROR: raw file not found at {RAW_PATH}", file=sys.stderr)
        return 1
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | seed {SEED}")
    print(f"raw: {RAW_PATH.relative_to(ROOT)}")

    # ---- 1. clean -----------------------------------------------------------
    t0 = time.perf_counter()
    clean_df, log_df = cl.clean(RAW_PATH, verbose=False)
    cl.save_outputs(clean_df, log_df, ROOT)
    _stage(
        1, "clean", t0,
        f"{log_df.loc[1, 'rows_before']:,} raw rows -> {len(clean_df):,} clean rows "
        f"({int(log_df['rows_removed'].sum()):,} removed over {len(log_df) - 1} rules); "
        f"{clean_df['Customer ID'].nunique():,} customers, {clean_df['StockCode'].nunique():,} products "
        f"-> clean.parquet, cleaning_log.csv",
    )

    # ---- 2. split -----------------------------------------------------------
    t0 = time.perf_counter()
    split_res = sp_.split(clean_df, months=sp_.TEST_MONTHS)
    sp_.leakage_checks(split_res)
    sp_.save_outputs(split_res, ROOT)
    s = split_res.stats
    _stage(
        2, "split", t0,
        f"cutoff {s['cutoff'].date()} | train {s['train_rows']:,} rows "
        f"({s['train_customers']:,} customers, {s['train_products']:,} products) | "
        f"test period {s['test_rows']:,} rows | cold-start excluded {s['cold_start_customers']:,} | "
        f"eval customers {s['eval_customers']:,}, targets {s['target_pairs']:,} "
        f"({s['targets_unseen_in_train']:,} unseen in train) | leakage checks passed "
        f"-> train.parquet, test_targets.parquet",
    )

    # ---- 3. matrices ---------------------------------------------------------
    t0 = time.perf_counter()
    mres = bm.build_matrix(split_res.train, value=VALUE, min_customers=MIN_CUSTOMERS)
    m = mres.stats
    report = bm.matrix_report(mres.A, f"A_{VALUE}")
    report.to_csv(REPORTS / "matrix_report.csv", index=False)
    _stage(
        3, "build_matrix", t0,
        f"{m['pairs_aggregated']:,} pairs -> {m['pairs_after_product_filter']:,} after min_customers={MIN_CUSTOMERS} "
        f"({m['products_before']:,} -> {m['products_after']:,} products, "
        f"{m['customers_before']:,} -> {m['customers_after']:,} customers) | "
        f"log/binary/raw built, nnz {m['nnz']:,} -> A_*.npz, customer_index, product_index, matrix_report.csv",
    )

    # ---- 4. flag targets -----------------------------------------------------
    t0 = time.perf_counter()
    targets_flagged, t = bm.flag_targets(split_res.targets, mres)
    bm.save_outputs(mres, targets_flagged, ROOT)
    _stage(
        4, "flag_targets", t0,
        f"{t['targets_total']:,} targets, {t['targets_in_matrix']:,} in_matrix ({t['targets_in_matrix_pct']:.2f}%) | "
        f"{t['eval_customers_with_in_matrix_target']:,} of {t['eval_customers']:,} eval customers reachable "
        f"(ceiling {t['hit_rate_ceiling_pct']:.2f}%) -> test_targets.parquet (+in_matrix)",
    )

    # ---- summary -------------------------------------------------------------
    A = mres.A
    sparsity = 100 * (1 - A.nnz / (A.shape[0] * A.shape[1]))
    print()
    print(f"A_{VALUE}: shape {A.shape[0]:,} x {A.shape[1]:,} | nnz {A.nnz:,} | sparsity {sparsity:.2f}%")
    print(f"total {time.perf_counter() - t_all:.1f}s | outputs in {PROCESSED.relative_to(ROOT)}/ and {REPORTS.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
