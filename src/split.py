"""DE3 - time-based train/test split with leakage checks.

Usage (from the repo root)::

    from src.split import split, leakage_checks, top_train_products, save_outputs
    result = split(clean_df)
    leakage_checks(result)

or as a script::

    python -m src.split

The split is purely temporal: the last ``months`` months of the cleaned data form the
test period. Test *targets* are the (customer, product) pairs bought in the test period
by a customer who exists in train and who never bought that product in train. No product
popularity filter is applied here (that is DE4, on training data only).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TEST_MONTHS = 2
CUSTOMER = "Customer ID"
PRODUCT = "StockCode"
DATE = "InvoiceDate"


@dataclass
class SplitResult:
    cutoff: pd.Timestamp
    train: pd.DataFrame
    test_period: pd.DataFrame
    targets: pd.DataFrame  # columns: Customer ID, StockCode, seen_in_train
    eval_customers: np.ndarray  # sorted
    cold_start_customers: np.ndarray  # in test period, not in train (excluded)
    repeat_only_customers: np.ndarray  # in train and test, but every test purchase is a repeat (no targets)
    stats: dict = field(default_factory=dict)


def compute_cutoff(df: pd.DataFrame, months: int = TEST_MONTHS) -> pd.Timestamp:
    """``(max InvoiceDate - months) `` normalised to midnight."""
    return (df[DATE].max() - pd.DateOffset(months=months)).normalize()


def split(clean_df: pd.DataFrame, months: int = TEST_MONTHS) -> SplitResult:
    df = clean_df
    cutoff = compute_cutoff(df, months)

    # 1. temporal split -------------------------------------------------------
    train = df.loc[df[DATE] < cutoff].reset_index(drop=True)
    test_period = df.loc[df[DATE] >= cutoff].reset_index(drop=True)

    # 2. targets = test pairs never bought by that customer in train ----------
    train_pairs = train[[CUSTOMER, PRODUCT]].drop_duplicates()
    test_pairs = test_period[[CUSTOMER, PRODUCT]].drop_duplicates()
    merged = test_pairs.merge(train_pairs, on=[CUSTOMER, PRODUCT], how="left", indicator=True)
    new_pairs = merged.loc[merged["_merge"] == "left_only", [CUSTOMER, PRODUCT]]
    n_repeat_pairs = int((merged["_merge"] == "both").sum())

    # 3. evaluation customers = in train AND have >= 1 target -----------------
    train_customers = np.sort(train[CUSTOMER].unique())
    test_customers = np.sort(test_period[CUSTOMER].unique())
    in_train = np.isin(test_customers, train_customers)
    cold_start = test_customers[~in_train]

    targets = new_pairs.loc[new_pairs[CUSTOMER].isin(train_customers)].copy()
    eval_customers = np.sort(targets[CUSTOMER].unique())
    repeat_only = np.setdiff1d(test_customers[in_train], eval_customers)

    # 4. flag whether anyone bought the target product in train ---------------
    train_products = train[PRODUCT].unique()
    targets["seen_in_train"] = targets[PRODUCT].isin(train_products)
    targets = targets.sort_values([CUSTOMER, PRODUCT]).reset_index(drop=True)

    per_customer = targets.groupby(CUSTOMER).size()
    stats = {
        "cutoff": cutoff,
        "train_rows": len(train),
        "train_date_min": train[DATE].min(),
        "train_date_max": train[DATE].max(),
        "train_customers": len(train_customers),
        "train_products": len(train_products),
        "test_rows": len(test_period),
        "test_date_min": test_period[DATE].min(),
        "test_date_max": test_period[DATE].max(),
        "test_customers": len(test_customers),
        "test_pairs_total": len(test_pairs),
        "test_pairs_repeat": n_repeat_pairs,
        "cold_start_customers": len(cold_start),
        "cold_start_pairs_dropped": int(new_pairs[CUSTOMER].isin(cold_start).sum()),
        "repeat_only_customers": len(repeat_only),
        "eval_customers": len(eval_customers),
        "target_pairs": len(targets),
        "targets_per_customer_mean": float(per_customer.mean()),
        "targets_per_customer_median": float(per_customer.median()),
        "targets_unseen_in_train": int((~targets["seen_in_train"]).sum()),
        "targets_unseen_in_train_pct": float((~targets["seen_in_train"]).mean() * 100),
        "eval_customers_all_unseen": int((~targets.groupby(CUSTOMER)["seen_in_train"].any()).sum()),
    }

    return SplitResult(
        cutoff=cutoff,
        train=train,
        test_period=test_period,
        targets=targets,
        eval_customers=eval_customers,
        cold_start_customers=cold_start,
        repeat_only_customers=repeat_only,
        stats=stats,
    )


def leakage_checks(r: SplitResult) -> list[str]:
    """Raise AssertionError on any leakage; return the list of checks that passed."""
    passed = []

    assert r.train[DATE].max() < r.cutoff <= r.test_period[DATE].min(), "date boundary violated"
    passed.append(f"train max {r.train[DATE].max()} < cutoff {r.cutoff} <= test min {r.test_period[DATE].min()}")

    train_pairs = r.train[[CUSTOMER, PRODUCT]].drop_duplicates()
    overlap = r.targets[[CUSTOMER, PRODUCT]].merge(train_pairs, on=[CUSTOMER, PRODUCT], how="inner")
    assert overlap.empty, f"{len(overlap)} target pairs exist in train"
    passed.append("no (customer, product) target pair exists in train")

    train_customers = r.train[CUSTOMER].unique()
    assert np.isin(r.eval_customers, train_customers).all(), "evaluation customer missing from train"
    passed.append("every evaluation customer exists in train")

    assert np.isin(r.targets[CUSTOMER].unique(), r.eval_customers).all()
    assert not r.targets.duplicated([CUSTOMER, PRODUCT]).any(), "duplicate target pairs"
    passed.append("targets are unique pairs and belong only to evaluation customers")

    assert not np.isin(r.cold_start_customers, train_customers).any()
    passed.append("cold-start customers are absent from train")
    return passed


def top_train_products(train: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Best-selling products in TRAIN by number of distinct customers (baseline input).

    Description = most frequent description for that code in train.
    """
    customers = train.groupby(PRODUCT)[CUSTOMER].nunique().rename("customers")
    desc = train.groupby(PRODUCT)["Description"].agg(lambda s: s.value_counts().index[0])
    out = pd.DataFrame({"customers": customers, "Description": desc}).sort_values(
        ["customers", PRODUCT], ascending=[False, True]
    )
    out = out.head(n).reset_index()[[PRODUCT, "Description", "customers"]]
    out.index = range(1, len(out) + 1)
    out.index.name = "rank"
    return out


def save_outputs(r: SplitResult, root: str | Path) -> tuple[Path, Path]:
    root = Path(root)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    targets_path = out_dir / "test_targets.parquet"
    r.train.to_parquet(train_path, index=False)
    r.targets[[CUSTOMER, PRODUCT, "seen_in_train"]].to_parquet(targets_path, index=False)
    return train_path, targets_path


def print_report(r: SplitResult) -> None:
    s = r.stats
    print(f"cutoff                      : {s['cutoff']}  (max InvoiceDate - {TEST_MONTHS} months, normalised)")
    print()
    print(f"TRAIN rows                  : {s['train_rows']:,}")
    print(f"      date range            : {s['train_date_min']}  ->  {s['train_date_max']}")
    print(f"      customers             : {s['train_customers']:,}")
    print(f"      products              : {s['train_products']:,}")
    print()
    print(f"TEST PERIOD rows            : {s['test_rows']:,}")
    print(f"      date range            : {s['test_date_min']}  ->  {s['test_date_max']}")
    print(f"      customers             : {s['test_customers']:,}")
    print(f"      distinct (cust, prod) : {s['test_pairs_total']:,}  of which repeat purchases: {s['test_pairs_repeat']:,}")
    print()
    print(f"cold-start customers excluded (in test, not in train) : {s['cold_start_customers']:,}"
          f"  ({s['cold_start_pairs_dropped']:,} new pairs dropped with them)")
    print(f"repeat-only customers excluded (in train, no new pair): {s['repeat_only_customers']:,}")
    print()
    print(f"evaluation customers        : {s['eval_customers']:,}")
    print(f"total target pairs          : {s['target_pairs']:,}")
    print(f"targets per eval customer   : mean {s['targets_per_customer_mean']:.2f}, median {s['targets_per_customer_median']:.0f}")
    print()
    print(f"target pairs with product never seen in train : {s['targets_unseen_in_train']:,} "
          f"({s['targets_unseen_in_train_pct']:.2f}%)  <- unwinnable for any model trained on train")
    print(f"evaluation customers whose targets are ALL unseen : {s['eval_customers_all_unseen']:,}")


def _main() -> int:
    root = Path(__file__).resolve().parents[1]
    clean_df = pd.read_parquet(root / "data" / "processed" / "clean.parquet")
    r = split(clean_df)
    for msg in leakage_checks(r):
        print("PASS:", msg)
    print()
    print_report(r)
    print()
    print(top_train_products(r.train).to_string())
    paths = save_outputs(r, root)
    print("saved:", *paths)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
