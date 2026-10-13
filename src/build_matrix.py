"""DE4/DE5 - sparse customer x product matrix from TRAINING data only.

Usage (from the repo root)::

    from src.build_matrix import build_matrix, flag_targets, matrix_report, save_outputs
    result = build_matrix(train_df, value="log", min_customers=10)

or as a script::

    python -m src.build_matrix

Order of operations (fixed, see decision_log.md):

1. aggregate Quantity per (Customer ID, StockCode) -> total_qty      (BEFORE any transform)
2. keep products bought by >= ``min_customers`` distinct customers
3. drop customers left with no rows
4. value transform: log1p(total_qty) / 1.0 / total_qty  (all three built)
5. sorted index maps  (row = anonymised customer index, col = product index)
6. csr_matrix with nnz asserted == number of aggregated pairs

``build_matrix`` never touches test data. Flagging test targets against the final
index maps is a separate step (``flag_targets``) that the notebook runs afterwards.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CUSTOMER = "Customer ID"
PRODUCT = "StockCode"
QTY = "Quantity"
VALUES = ("log", "binary", "raw")


@dataclass
class MatrixResult:
    value: str  # which of the three is the "main" matrix (``A``)
    min_customers: int
    matrices: dict  # {"log": csr, "binary": csr, "raw": csr}
    customer_index: pd.DataFrame  # Customer ID, row
    product_index: pd.DataFrame  # StockCode, col, Description, n_customers
    pairs: pd.DataFrame  # aggregated + filtered: Customer ID, StockCode, total_qty, row, col
    stats: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    @property
    def A(self) -> sp.csr_matrix:
        return self.matrices[self.value]


def _transform(total_qty: np.ndarray, value: str) -> np.ndarray:
    if value == "log":
        return np.log1p(total_qty.astype(np.float64))
    if value == "binary":
        return np.ones(len(total_qty), dtype=np.float64)
    if value == "raw":
        return total_qty.astype(np.float64)
    raise ValueError(f"value must be one of {VALUES}, got {value!r}")


def build_matrix(train_df: pd.DataFrame, value: str = "log", min_customers: int = 10) -> MatrixResult:
    if value not in VALUES:
        raise ValueError(f"value must be one of {VALUES}, got {value!r}")
    if min_customers < 1:
        raise ValueError("min_customers must be >= 1")

    steps: list[str] = []
    stats: dict = {"value": value, "min_customers": min_customers}
    stats["train_rows"] = len(train_df)
    stats["train_customers"] = int(train_df[CUSTOMER].nunique())
    stats["train_products"] = int(train_df[PRODUCT].nunique())

    # 1. aggregate per pair BEFORE any transform ------------------------------
    pairs = (
        train_df.groupby([CUSTOMER, PRODUCT], sort=False, as_index=False)[QTY]
        .sum()
        .rename(columns={QTY: "total_qty"})
    )
    assert not pairs.duplicated([CUSTOMER, PRODUCT]).any()
    assert (pairs["total_qty"] > 0).all(), "aggregated quantities must be positive"
    stats["pairs_aggregated"] = len(pairs)
    steps.append(
        f"[1] aggregate      : {len(train_df):,} train lines -> {len(pairs):,} (customer, product) pairs; "
        f"{stats['train_customers']:,} customers x {stats['train_products']:,} products"
    )

    # 2. product filter (rows per product == distinct customers) --------------
    per_product = pairs[PRODUCT].value_counts()
    keep_products = per_product.index[per_product >= min_customers]
    before_pairs, before_products = len(pairs), len(per_product)
    pairs = pairs.loc[pairs[PRODUCT].isin(keep_products)]
    stats.update(
        products_before=before_products,
        products_after=len(keep_products),
        products_dropped=before_products - len(keep_products),
        pairs_removed_by_product_filter=before_pairs - len(pairs),
        pairs_after_product_filter=len(pairs),
    )
    steps.append(
        f"[2] product filter : keep products with >= {min_customers} customers: "
        f"{before_products:,} -> {len(keep_products):,} products "
        f"({stats['products_dropped']:,} dropped); pairs {before_pairs:,} -> {len(pairs):,} "
        f"({stats['pairs_removed_by_product_filter']:,} removed)"
    )

    # 3. drop customers with no rows left --------------------------------------
    customers_before = train_df[CUSTOMER].unique()
    customers_after = pairs[CUSTOMER].unique()
    dropped_customers = np.sort(np.setdiff1d(customers_before, customers_after))
    stats.update(
        customers_before=len(customers_before),
        customers_after=len(customers_after),
        customers_dropped=len(dropped_customers),
    )
    steps.append(
        f"[3] empty customers: {len(customers_before):,} -> {len(customers_after):,} customers "
        f"({len(dropped_customers):,} dropped, every product they bought fell under the filter)"
    )

    # 5. sorted index maps -----------------------------------------------------
    cust_sorted = np.sort(customers_after)
    prod_sorted = np.sort(pairs[PRODUCT].unique())
    customer_index = pd.DataFrame({CUSTOMER: cust_sorted, "row": np.arange(len(cust_sorted), dtype=np.int64)})
    row_of = pd.Series(customer_index["row"].to_numpy(), index=cust_sorted)
    col_of = pd.Series(np.arange(len(prod_sorted), dtype=np.int64), index=prod_sorted)

    pairs = pairs.assign(
        row=row_of.reindex(pairs[CUSTOMER]).to_numpy(),
        col=col_of.reindex(pairs[PRODUCT]).to_numpy(),
    ).sort_values(["row", "col"]).reset_index(drop=True)

    n_customers_per_product = pairs.groupby(PRODUCT).size()
    if "Description" in train_df.columns:
        desc = (
            train_df.loc[train_df[PRODUCT].isin(prod_sorted)]
            .groupby(PRODUCT)["Description"]
            .agg(lambda s: s.value_counts().index[0])
        )
    else:
        desc = pd.Series(pd.NA, index=prod_sorted, dtype="str")
    product_index = pd.DataFrame(
        {
            PRODUCT: prod_sorted,
            "col": np.arange(len(prod_sorted), dtype=np.int64),
            "Description": desc.reindex(prod_sorted).to_numpy(),
            "n_customers": n_customers_per_product.reindex(prod_sorted).to_numpy().astype(np.int64),
        }
    )
    assert (product_index["n_customers"] >= min_customers).all()
    steps.append(
        f"[5] index maps     : {len(customer_index):,} customers -> rows 0..{len(customer_index) - 1}; "
        f"{len(product_index):,} products -> cols 0..{len(product_index) - 1}"
    )

    # 4 + 6. values and csr matrices -------------------------------------------
    shape = (len(customer_index), len(product_index))
    rows = pairs["row"].to_numpy()
    cols = pairs["col"].to_numpy()
    total_qty = pairs["total_qty"].to_numpy()

    matrices = {}
    for v in VALUES:
        A = sp.csr_matrix((_transform(total_qty, v), (rows, cols)), shape=shape, dtype=np.float64)
        A.sum_duplicates()
        A.sort_indices()
        assert A.nnz == len(pairs), f"{v}: nnz {A.nnz:,} != pairs {len(pairs):,} (duplicates were summed!)"
        matrices[v] = A

    ref = matrices[value]
    for v, A in matrices.items():
        assert A.shape == ref.shape, f"{v}: shape mismatch"
        assert np.array_equal(A.indptr, ref.indptr) and np.array_equal(A.indices, ref.indices), f"{v}: sparsity pattern mismatch"
    stats.update(shape=shape, nnz=int(ref.nnz))
    steps.append(
        f"[4/6] matrices     : built log/binary/raw as csr float64, shape {shape[0]:,} x {shape[1]:,}, "
        f"nnz {ref.nnz:,} == {len(pairs):,} pairs; identical shape and sparsity pattern"
    )

    return MatrixResult(
        value=value,
        min_customers=min_customers,
        matrices=matrices,
        customer_index=customer_index,
        product_index=product_index,
        pairs=pairs,
        stats=stats,
        steps=steps,
    )


# --------------------------------------------------------------------------- #
# Step 7 - test targets against the final index maps (NOT used when building)
# --------------------------------------------------------------------------- #
def flag_targets(targets: pd.DataFrame, result: MatrixResult) -> tuple[pd.DataFrame, dict]:
    """Add ``in_matrix`` to the test targets and compute the hit-rate ceiling."""
    out = targets.copy()
    in_cust = out[CUSTOMER].isin(result.customer_index[CUSTOMER])
    in_prod = out[PRODUCT].isin(result.product_index[PRODUCT])
    out["in_matrix"] = (in_cust & in_prod).to_numpy()

    eval_customers = out[CUSTOMER].unique()
    train_customers_all = result.stats["customers_before"]
    stats = {
        "targets_total": len(out),
        "targets_in_matrix": int(out["in_matrix"].sum()),
        "targets_in_matrix_pct": float(out["in_matrix"].mean() * 100),
        "targets_customer_not_in_matrix": int((~in_cust).sum()),
        "targets_product_not_in_matrix": int((~in_prod).sum()),
        "eval_customers": len(eval_customers),
        "eval_customers_in_matrix": int(np.isin(eval_customers, result.customer_index[CUSTOMER]).sum()),
        "eval_customers_with_in_matrix_target": int(out.groupby(CUSTOMER)["in_matrix"].any().sum()),
        "train_customers_all": train_customers_all,
    }
    stats["eval_customers_dropped_by_filter"] = stats["eval_customers"] - stats["eval_customers_in_matrix"]
    stats["hit_rate_ceiling_pct"] = stats["eval_customers_with_in_matrix_target"] / stats["eval_customers"] * 100
    return out, stats


# --------------------------------------------------------------------------- #
# DE5 - matrix report
# --------------------------------------------------------------------------- #
def matrix_report(A: sp.csr_matrix, name: str = "A_log") -> pd.DataFrame:
    n_rows, n_cols = A.shape
    dense_mb = n_rows * n_cols * 8 / 1e6
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    nnz_row = np.diff(A.indptr)
    nnz_col = np.bincount(A.indices, minlength=n_cols)
    vals = A.data

    rows = [
        ("matrix", name),
        ("rows (customers)", n_rows),
        ("cols (products)", n_cols),
        ("nnz", int(A.nnz)),
        ("sparsity_pct", 100 * (1 - A.nnz / (n_rows * n_cols))),
        ("dense_memory_mb", dense_mb),
        ("sparse_memory_mb", sparse_mb),
        ("compression_ratio_dense_over_sparse", dense_mb / sparse_mb),
        ("value_min", float(vals.min())),
        ("value_median", float(np.median(vals))),
        ("value_max", float(vals.max())),
        ("nnz_per_row_min (products per customer)", int(nnz_row.min())),
        ("nnz_per_row_median", float(np.median(nnz_row))),
        ("nnz_per_row_max", int(nnz_row.max())),
        ("nnz_per_col_min (customers per product)", int(nnz_col.min())),
        ("nnz_per_col_median", float(np.median(nnz_col))),
        ("nnz_per_col_max", int(nnz_col.max())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# --------------------------------------------------------------------------- #
# Step 8 - save
# --------------------------------------------------------------------------- #
def save_outputs(result: MatrixResult, targets_flagged: pd.DataFrame, root: str | Path) -> list[Path]:
    root = Path(root)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for v, A in result.matrices.items():
        p = out_dir / f"A_{v}.npz"
        sp.save_npz(p, A)
        written.append(p)
    p = out_dir / "customer_index.parquet"
    result.customer_index.to_parquet(p, index=False)
    written.append(p)
    p = out_dir / "product_index.parquet"
    result.product_index.to_parquet(p, index=False)
    written.append(p)
    p = out_dir / "test_targets.parquet"
    targets_flagged.to_parquet(p, index=False)
    written.append(p)
    return written


def _main() -> int:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    result = build_matrix(train, value="log", min_customers=10)
    print("\n".join(result.steps))
    targets = pd.read_parquet(root / "data" / "processed" / "test_targets.parquet")
    flagged, tstats = flag_targets(targets, result)
    for k, v in tstats.items():
        print(f"{k:>40}: {v}")
    report = matrix_report(result.A, "A_log")
    print(report.to_string(index=False))
    report.to_csv(root / "reports" / "matrix_report.csv", index=False)
    for p in save_outputs(result, flagged, root):
        print("saved:", p.relative_to(root))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
