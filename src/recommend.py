"""ML3 - recommendation functions on top of the Truncated SVD factors.

Usage (from the repo root)::

    from src.recommend import fit_recommender, similar_products, precompute_item_neighbors, \
        recommend_for_customers, best_seller_recs

    rec = fit_recommender(A_log, product_index, k=50, weighting="V")
    idx, scores = similar_products(rec, j=123, n=10)
    neighbors, secs = precompute_item_neighbors(rec, n=10)
    recs, secs = recommend_for_customers(rec, rows=np.arange(A_log.shape[0]), n=10)

Product vectors
---------------
``weighting="V"``      : P = components_.T                        (brief's starter code)
``weighting="VSigma"`` : P = components_.T * singular_values_     (weights leading components more)

Rows of P are L2-normalised, so ``P @ P.T`` is the cosine-similarity matrix. It is never
materialised in full here: item neighbours are computed in row batches and customer scores
go through the k-dimensional space, ``(A[rows] @ P) @ P.T``.

Nothing in this module reads test data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from src.svd import N_ITER, SEED, fit_svd

WEIGHTINGS = ("V", "VSigma")


# --------------------------------------------------------------------------- #
# Product vectors
# --------------------------------------------------------------------------- #
def product_vectors(svd: TruncatedSVD, weighting: str = "V") -> tuple[np.ndarray, np.ndarray]:
    """Return (P, zero_norm_cols). P has unit-length rows; all-zero rows stay zero (no NaN)."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    V = svd.components_.T.astype(np.float64, copy=True)  # (m, k)
    if weighting == "VSigma":
        V *= svd.singular_values_[np.newaxis, :]
    norms = np.linalg.norm(V, axis=1)
    zero_norm = np.flatnonzero(norms == 0)
    P = normalize(V, norm="l2", axis=1, copy=True)  # leaves zero rows as zeros
    return P, zero_norm


@dataclass
class Recommender:
    svd: TruncatedSVD
    k: int
    weighting: str
    P: np.ndarray  # (m, k) unit-norm product vectors
    A: sp.csr_matrix  # training matrix (customers x products)
    product_index: pd.DataFrame
    zero_norm_cols: np.ndarray
    fit_seconds: float
    meta: dict = field(default_factory=dict)

    @property
    def n_customers(self) -> int:
        return self.A.shape[0]

    @property
    def n_products(self) -> int:
        return self.A.shape[1]


def fit_recommender(
    A: sp.csr_matrix,
    product_index: pd.DataFrame,
    k: int = 50,
    weighting: str = "V",
    seed: int = SEED,
    n_iter: int = N_ITER,
) -> Recommender:
    A = sp.csr_matrix(A)
    assert A.shape[1] == len(product_index), "product_index does not match A"
    svd, secs = fit_svd(A, k, seed=seed, n_iter=n_iter)
    P, zero_norm = product_vectors(svd, weighting)
    return Recommender(
        svd=svd, k=k, weighting=weighting, P=P, A=A, product_index=product_index,
        zero_norm_cols=zero_norm, fit_seconds=secs,
    )


def with_weighting(rec: Recommender, weighting: str) -> Recommender:
    """Same fitted SVD, different product-vector weighting (no refit)."""
    P, zero_norm = product_vectors(rec.svd, weighting)
    return Recommender(
        svd=rec.svd, k=rec.k, weighting=weighting, P=P, A=rec.A, product_index=rec.product_index,
        zero_norm_cols=zero_norm, fit_seconds=rec.fit_seconds,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def top_n_from_scores(S: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row top-n column indices (descending score) and the scores. S is (b, m)."""
    n = min(n, S.shape[1])
    part = np.argpartition(-S, n - 1, axis=1)[:, :n]
    part_scores = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.take_along_axis(part_scores, order, axis=1)


def _batches(n_total: int, batch_size: int):
    for start in range(0, n_total, batch_size):
        yield start, min(start + batch_size, n_total)


# --------------------------------------------------------------------------- #
# Item-item
# --------------------------------------------------------------------------- #
def similar_products(rec: Recommender, j: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Top-n products most similar to product column j by cosine similarity, excluding j."""
    sims = rec.P @ rec.P[j]  # (m,)
    sims[j] = -np.inf
    idx, scores = top_n_from_scores(sims[np.newaxis, :], n)
    return idx[0], scores[0]


def precompute_item_neighbors(rec: Recommender, n: int = 10, batch_size: int = 512) -> tuple[pd.DataFrame, float]:
    """Nightly 'customers who bought this also bought' table for every product.

    Computed in row batches of P @ P.T (batch x m, float64), never the full m x m matrix.
    """
    t0 = time.perf_counter()
    m = rec.n_products
    all_idx = np.empty((m, n), dtype=np.int64)
    all_scores = np.empty((m, n), dtype=np.float64)
    for s, e in _batches(m, batch_size):
        S = rec.P[s:e] @ rec.P.T  # (b, m)
        S[np.arange(e - s), np.arange(s, e)] = -np.inf  # exclude self
        idx, sc = top_n_from_scores(S, n)
        all_idx[s:e] = idx
        all_scores[s:e] = sc
    secs = time.perf_counter() - t0

    codes = rec.product_index["StockCode"].to_numpy()
    descs = rec.product_index["Description"].to_numpy()
    rows = np.repeat(np.arange(m), n)
    ranks = np.tile(np.arange(1, n + 1), m)
    flat_idx = all_idx.ravel()
    table = pd.DataFrame(
        {
            "product_col": rows,
            "StockCode": codes[rows],
            "Description": descs[rows],
            "rank": ranks,
            "neighbor_col": flat_idx,
            "neighbor_StockCode": codes[flat_idx],
            "neighbor_Description": descs[flat_idx],
            "score": all_scores.ravel(),
        }
    )
    return table, secs


# --------------------------------------------------------------------------- #
# Customer recommendations
# --------------------------------------------------------------------------- #
def _mask_bought(S: np.ndarray, A_block: sp.csr_matrix) -> None:
    """In-place: set scores of products already bought (non-zero in A_block) to -inf."""
    r, c = A_block.nonzero()
    S[r, c] = -np.inf


def recommend_for_customers(
    rec: Recommender, rows: np.ndarray, n: int = 10, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray, float]:
    """Top-n product columns per customer row.

    scores = A[rows] @ P @ P.T computed as (A[rows] @ P) @ P.T  ->  (b, k) then (b, m).
    Products already bought in train are set to -inf before taking the top n.
    Returns (idx (len(rows), n), scores, seconds).
    """
    t0 = time.perf_counter()
    rows = np.asarray(rows)
    out_idx = np.empty((len(rows), n), dtype=np.int64)
    out_scores = np.empty((len(rows), n), dtype=np.float64)
    for s, e in _batches(len(rows), batch_size):
        A_b = rec.A[rows[s:e]]  # (b, m) sparse
        Z = A_b @ rec.P  # (b, k)
        S = Z @ rec.P.T  # (b, m)
        _mask_bought(S, A_b)
        idx, sc = top_n_from_scores(S, n)
        out_idx[s:e] = idx
        out_scores[s:e] = sc
    return out_idx, out_scores, time.perf_counter() - t0


def best_seller_recs(
    rec: Recommender, rows: np.ndarray, n: int = 10, batch_size: int = 2048
) -> tuple[np.ndarray, float]:
    """Baseline: top products by distinct TRAIN customers (product_index.n_customers),
    skipping products the customer already bought. Ties broken by column order."""
    t0 = time.perf_counter()
    pop = rec.product_index["n_customers"].to_numpy()
    order = np.lexsort((np.arange(len(pop)), -pop))  # best seller first, then col asc
    rank_score = np.empty(len(pop), dtype=np.float64)
    rank_score[order] = -np.arange(len(pop), dtype=np.float64)  # 0 for the best seller, -1 next, ...
    rows = np.asarray(rows)
    out = np.empty((len(rows), n), dtype=np.int64)
    for s, e in _batches(len(rows), batch_size):
        A_b = rec.A[rows[s:e]]
        S = np.broadcast_to(rank_score, (e - s, len(pop))).copy()
        _mask_bought(S, A_b)
        out[s:e], _ = top_n_from_scores(S, n)
    return out, time.perf_counter() - t0


def best_seller_order(rec: Recommender) -> np.ndarray:
    """Global best-seller column order (no customer exclusion)."""
    pop = rec.product_index["n_customers"].to_numpy()
    return np.lexsort((np.arange(len(pop)), -pop))


# --------------------------------------------------------------------------- #
# Inspection helpers (notebook)
# --------------------------------------------------------------------------- #
def describe_cols(rec: Recommender, cols: np.ndarray, scores: np.ndarray | None = None) -> pd.DataFrame:
    pi = rec.product_index
    out = pd.DataFrame(
        {
            "col": cols,
            "StockCode": pi["StockCode"].to_numpy()[cols],
            "Description": pi["Description"].to_numpy()[cols],
            "n_customers_train": pi["n_customers"].to_numpy()[cols],
        }
    )
    if scores is not None:
        out["score"] = scores
    out.index = range(1, len(out) + 1)
    out.index.name = "rank"
    return out


def customer_top_bought(rec: Recommender, row: int, n: int = 5) -> pd.DataFrame:
    """The n products with the largest A_log value for an anonymised customer row."""
    r = rec.A.getrow(row)
    cols, vals = r.indices, r.data
    order = np.argsort(-vals, kind="stable")[:n]
    out = describe_cols(rec, cols[order])
    out["A_log_value"] = vals[order]
    out["approx_total_qty"] = np.expm1(vals[order]).round().astype(int)
    return out


def find_col_by_description(rec: Recommender, description: str) -> int:
    """Column of the product whose Description matches exactly (most customers if several)."""
    pi = rec.product_index
    hits = pi.loc[pi["Description"].str.strip() == description.strip()]
    if hits.empty:
        raise KeyError(description)
    return int(hits.sort_values("n_customers", ascending=False)["col"].iloc[0])


def assert_no_bought_in_recs(rec: Recommender, rows: np.ndarray, recs: np.ndarray) -> None:
    """Vectorised check that no recommended product was already bought by that customer."""
    A_rows = rec.A[np.asarray(rows)]
    r = np.repeat(np.arange(len(rows)), recs.shape[1])
    bought = np.asarray(A_rows[r, recs.ravel()]).ravel()
    assert not np.any(bought != 0), f"{int(np.sum(bought != 0))} recommendations were already bought"


def save_product_factors(rec: Recommender, root: str | Path) -> Path:
    root = Path(root)
    path = root / "data" / "processed" / f"product_factors_k{rec.k}.npy"
    np.save(path, rec.P)
    return path
