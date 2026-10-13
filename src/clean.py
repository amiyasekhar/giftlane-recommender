"""DE2 - cleaning pipeline for UCI Online Retail II with an auditable cleaning log.

Usage (from the repo root)::

    from src.clean import clean
    clean_df, log_df = clean("data/raw/online_retail_II.xlsx")

or as a script::

    python -m src.clean

Every rule is applied in a fixed order and each dropped row is counted under the
*first* rule that removes it. Nothing in ``data/raw/`` is touched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SHEET_OLD = "Year 2009-2010"
SHEET_NEW = "Year 2010-2011"
EXPECTED_RAW_ROWS = 1_067_371
EXPECTED_OVERLAP_ROWS = 22_523

STANDARD_CODE = re.compile(r"^\d{5}[A-Z]*$")

# Non-product codes (fees, postage, adjustments, vouchers, test rows) to drop in rule 6.
NON_PRODUCT_CODES = frozenset(
    {"POST", "DOT", "M", "D", "S", "B", "C2", "BANK CHARGES", "ADJUST", "AMAZONFEE", "CRUK", "TEST001"}
)
NON_PRODUCT_PREFIXES = ("GIFT_",)

LOG_COLUMNS = ["step", "rule", "rows_before", "rows_removed", "rows_after", "reason"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 0 - type normalisation only, no rows removed.

    * ``StockCode``  -> str, stripped, upper-cased  (raw column mixes int and str)
    * ``Invoice``    -> str, stripped               (raw column mixes int and str)
    * ``Description``-> str, NaN preserved          (4 raw cells are ints; mixed object
      columns cannot be written to parquet)
    * ``Customer ID`` stays float64 until missing values are removed in rule 7.
    """
    out = df.copy()
    out["StockCode"] = out["StockCode"].astype("str").str.strip().str.upper()
    out["Invoice"] = out["Invoice"].astype("str").str.strip()
    out["Description"] = out["Description"].astype("str")
    return out


def is_non_product(code: pd.Series) -> pd.Series:
    """Boolean mask of StockCodes on the explicit exclusion list (rule 6)."""
    return code.isin(NON_PRODUCT_CODES) | code.str.startswith(NON_PRODUCT_PREFIXES)


def non_standard_stockcodes(df: pd.DataFrame) -> pd.DataFrame:
    """All StockCodes that do not match ``^\\d{5}[A-Z]*$`` with row count, most common
    Description and the action rule 6 will take. Sorted by row count descending."""
    mask = ~df["StockCode"].str.fullmatch(STANDARD_CODE.pattern)
    sub = df.loc[mask, ["StockCode", "Description"]]
    if sub.empty:
        return pd.DataFrame(columns=["StockCode", "rows", "most_common_description", "action"])

    rows = sub["StockCode"].value_counts()
    desc = (
        sub.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.value_counts().index[0])
    )
    table = pd.DataFrame(
        {
            "StockCode": rows.index,
            "rows": rows.to_numpy(),
            "most_common_description": desc.reindex(rows.index).to_numpy(),
        }
    )
    table["action"] = ["drop (non-product)" if x else "keep" for x in is_non_product(table["StockCode"])]
    return table.reset_index(drop=True)


class _Log:
    """Collects one record per rule."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, rule: str, before: int, after: int, reason: str) -> None:
        self.rows.append(
            {
                "step": len(self.rows),
                "rule": rule,
                "rows_before": int(before),
                "rows_removed": int(before - after),
                "rows_after": int(after),
                "reason": reason,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=LOG_COLUMNS)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def clean(
    raw_path: str | Path,
    verbose: bool = True,
    details: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both sheets of the raw workbook and apply the DE2 cleaning rules in order.

    Parameters
    ----------
    raw_path : path to ``online_retail_II.xlsx``
    verbose  : print per-rule progress and the non-standard StockCode review tables
    details  : optional dict; if given it is filled with intermediate review tables
               (``"non_standard_stockcodes"``, ``"unlisted_non_standard"``) for display
               in a notebook.

    Returns
    -------
    (clean_df, log_df)
    """
    raw_path = Path(raw_path)
    log = _Log()
    say = print if verbose else (lambda *a, **k: None)

    sheets = pd.read_excel(raw_path, sheet_name=None)
    assert set(sheets) == {SHEET_OLD, SHEET_NEW}, f"unexpected sheets: {list(sheets)}"

    # ---- Rule 0: normalise types (no removal) ---------------------------------
    sheets = {name: normalise_types(sdf) for name, sdf in sheets.items()}
    n_raw = sum(len(s) for s in sheets.values())
    assert n_raw == EXPECTED_RAW_ROWS, f"rows before rule 1 = {n_raw:,}, expected {EXPECTED_RAW_ROWS:,}"
    log.add(
        "0 normalise types",
        n_raw,
        n_raw,
        "StockCode -> str/strip/upper; Invoice -> str/strip; Description -> str (NaN kept); "
        "Customer ID left as float64. No rows removed.",
    )
    say(f"[0] normalise types          : {n_raw:>9,} rows (no removal)")

    # ---- Rule 1: cross-sheet overlap ------------------------------------------
    old, new = sheets[SHEET_OLD], sheets[SHEET_NEW]
    cutoff = old["InvoiceDate"].max()
    overlap_mask = new["InvoiceDate"] <= cutoff
    n_overlap = int(overlap_mask.sum())
    assert n_overlap == EXPECTED_OVERLAP_ROWS, f"overlap rows = {n_overlap:,}, expected {EXPECTED_OVERLAP_ROWS:,}"
    new = new.loc[~overlap_mask]
    df = pd.concat([old, new], ignore_index=True)
    log.add(
        "1 cross-sheet overlap",
        n_raw,
        len(df),
        f"Dropped rows of '{SHEET_NEW}' with InvoiceDate <= {cutoff} (max of '{SHEET_OLD}'); "
        f"DE1 showed every row in that window is an exact copy of a row in the earlier sheet.",
    )
    say(f"[1] cross-sheet overlap      : -{n_overlap:>8,} -> {len(df):,}")

    # ---- Rule 2: exact duplicate rows -----------------------------------------
    before = len(df)
    df = df.loc[~df.duplicated(keep="first")]
    log.add(
        "2 exact duplicates",
        before,
        len(df),
        "Rows identical in every column (keep first occurrence); same line repeated inside one invoice is an export artefact.",
    )
    say(f"[2] exact duplicates         : -{before - len(df):>8,} -> {len(df):,}")

    # Count needed for rule 4's reason, measured *before* rule 3 removes 'C' invoices.
    is_cancel = df["Invoice"].str.startswith("C")
    n_nonpos_qty_not_cancel = int(((df["Quantity"] <= 0) & ~is_cancel).sum())

    # ---- Rule 3: cancellations ------------------------------------------------
    before = len(df)
    n_cancel_invoices = df.loc[is_cancel, "Invoice"].nunique()
    df = df.loc[~is_cancel]
    log.add(
        "3 cancellations",
        before,
        len(df),
        f"Invoice starts with 'C' ({n_cancel_invoices:,} distinct cancellation invoices). "
        "Removed without netting against the original purchase.",
    )
    say(f"[3] cancellations            : -{before - len(df):>8,} -> {len(df):,}")

    # ---- Rule 4: Quantity <= 0 -------------------------------------------------
    before = len(df)
    df = df.loc[df["Quantity"] > 0]
    log.add(
        "4 quantity <= 0",
        before,
        len(df),
        f"Non-positive quantity outside cancellation invoices (stock adjustments / damages). "
        f"{n_nonpos_qty_not_cancel:,} such rows were NOT on a 'C' invoice before rule 3 "
        f"(DE1 counted 3,457 on the raw data before overlap/duplicate removal).",
    )
    say(f"[4] quantity <= 0            : -{before - len(df):>8,} -> {len(df):,}")

    # ---- Rule 5: Price <= 0 ----------------------------------------------------
    before = len(df)
    n_zero = int((df["Price"] == 0).sum())
    n_neg = int((df["Price"] < 0).sum())
    df = df.loc[df["Price"] > 0]
    log.add(
        "5 price <= 0",
        before,
        len(df),
        f"Zero price ({n_zero:,} rows: free items / missing price) or negative price ({n_neg:,} rows); not a real purchase signal.",
    )
    say(f"[5] price <= 0               : -{before - len(df):>8,} -> {len(df):,}")

    # ---- Rule 6: non-product StockCodes ---------------------------------------
    review = non_standard_stockcodes(df)
    unlisted = review.loc[review["action"] == "keep"].reset_index(drop=True)
    if details is not None:
        details["non_standard_stockcodes"] = review
        details["unlisted_non_standard"] = unlisted

    if verbose:
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 60):
            print(f"\nAll StockCodes not matching ^\\d{{5}}[A-Z]*$ at this stage ({len(review)} codes):")
            print(review.to_string(index=False))
            missing_from_data = sorted(NON_PRODUCT_CODES - set(review["StockCode"]))
            if missing_from_data:
                print(f"\nExclusion-list codes with 0 rows left at this stage (already removed by rules 1-5): {missing_from_data}")
            print(f"\nNon-matching codes NOT on the exclusion list (kept; review needed) ({len(unlisted)} codes):")
            print(unlisted.to_string(index=False) if len(unlisted) else "(none)")
            print()

    before = len(df)
    drop_mask = is_non_product(df["StockCode"])
    dropped_codes = sorted(df.loc[drop_mask, "StockCode"].unique())
    df = df.loc[~drop_mask]
    log.add(
        "6 non-product stockcodes",
        before,
        len(df),
        f"Postage, fees, manual/adjustment lines, bank charges, commission, vouchers and test rows: "
        f"{dropped_codes}. DCGS* codes and PADS kept (real products).",
    )
    say(f"[6] non-product stockcodes   : -{before - len(df):>8,} -> {len(df):,}")

    # ---- Rule 7: missing Customer ID -----------------------------------------
    before = len(df)
    df = df.loc[df["Customer ID"].notna()].copy()
    df["Customer ID"] = df["Customer ID"].astype("int64")
    log.add(
        "7 missing customer id",
        before,
        len(df),
        "Rows without a customer cannot be placed in a customer x product matrix. Customer ID then cast to int64.",
    )
    say(f"[7] missing customer id      : -{before - len(df):>8,} -> {len(df):,}")

    df = df.reset_index(drop=True)
    log_df = log.frame()
    assert log_df["rows_before"].iloc[1] == EXPECTED_RAW_ROWS
    assert log_df["rows_after"].iloc[-1] == len(df)
    return df, log_df


def summarise(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "distinct_customers": int(df["Customer ID"].nunique()),
        "distinct_products": int(df["StockCode"].nunique()),
        "distinct_invoices": int(df["Invoice"].nunique()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
    }


def save_outputs(clean_df: pd.DataFrame, log_df: pd.DataFrame, root: str | Path) -> tuple[Path, Path]:
    root = Path(root)
    parquet_path = root / "data" / "processed" / "clean.parquet"
    log_path = root / "reports" / "cleaning_log.csv"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(parquet_path, index=False)
    log_df.to_csv(log_path, index=False)
    return parquet_path, log_path


def _main() -> int:
    root = Path(__file__).resolve().parents[1]
    clean_df, log_df = clean(root / "data" / "raw" / "online_retail_II.xlsx")
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(log_df.to_string(index=False))
    for k, v in summarise(clean_df).items():
        print(f"{k:>20}: {v:,}" if isinstance(v, int) else f"{k:>20}: {v}")
    pq, lg = save_outputs(clean_df, log_df, root)
    print("saved:", pq, "and", lg)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
