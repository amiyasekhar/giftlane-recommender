"""ML4 - evaluation of the recommenders on held-out purchases.

Usage (from the repo root)::

    from src.evaluate import build_eval_set, run_grid, evaluate_baseline, paired_bootstrap, ...

Everything here reuses ``src.recommend``; no recommender is re-implemented. Metrics are
computed per evaluation customer and then averaged:

    hit@10       = 1 if any target is in the top 10 else 0
    recall@10    = |top10 ∩ targets| / |targets|      (all targets, incl. unreachable ones)
    precision@10 = |top10 ∩ targets| / 10
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.recommend import (
    Recommender,
    best_seller_recs,
    fit_recommender,
    precompute_item_neighbors,
    recommend_for_customers,
    with_weighting,
)
from src.svd import N_ITER, SEED

TOP_N = 10
K_GRID = (20, 50, 100)
WEIGHTINGS = ("V", "VSigma")
ROW_SOURCES = ("log", "binary")
HISTORY_BUCKETS = [(1, 10), (11, 30), (31, 100), (101, np.inf)]
RARE_BAND = (10, 20)


# --------------------------------------------------------------------------- #
# Evaluation set
# --------------------------------------------------------------------------- #
@dataclass
class EvalSet:
    rows: np.ndarray  # matrix row per evaluation customer (n_eval,)
    T: sp.csr_matrix  # (n_eval, m) boolean: in-matrix targets
    n_targets: np.ndarray  # all targets per customer, incl. not-in-matrix (n_eval,)
    n_targets_in_matrix: np.ndarray
    reachable: np.ndarray  # bool (n_eval,): >= 1 in-matrix target
    n_products: int

    @property
    def n_eval(self) -> int:
        return len(self.rows)

    @property
    def ceiling_pct(self) -> float:
        return 100 * float(self.reachable.mean())


def build_eval_set(targets: pd.DataFrame, customer_index: pd.DataFrame, product_index: pd.DataFrame) -> EvalSet:
    row_of = pd.Series(customer_index["row"].to_numpy(), index=customer_index["Customer ID"].to_numpy())
    col_of = pd.Series(product_index["col"].to_numpy(), index=product_index["StockCode"].to_numpy())

    customers = np.sort(targets["Customer ID"].unique())
    assert np.isin(customers, row_of.index).all(), "an evaluation customer has no matrix row"
    rows = row_of.reindex(customers).to_numpy().astype(np.int64)
    eval_pos = pd.Series(np.arange(len(customers)), index=customers)

    n_targets = targets.groupby("Customer ID").size().reindex(customers).to_numpy()
    in_m = targets.loc[targets["in_matrix"]]
    r = eval_pos.reindex(in_m["Customer ID"]).to_numpy()
    c = col_of.reindex(in_m["StockCode"]).to_numpy()
    assert not np.isnan(c).any(), "an in_matrix target has no product column"
    T = sp.csr_matrix((np.ones(len(r), dtype=bool), (r, c.astype(np.int64))), shape=(len(customers), len(product_index)))
    T.sum_duplicates()
    n_in = np.diff(T.indptr)
    return EvalSet(
        rows=rows, T=T, n_targets=n_targets, n_targets_in_matrix=n_in,
        reachable=n_in > 0, n_products=len(product_index),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class PerCustomer:
    hits: np.ndarray  # |top10 ∩ targets| per customer
    hit: np.ndarray  # 0/1
    recall: np.ndarray
    precision: np.ndarray

    def summary(self, reachable: np.ndarray | None = None) -> dict:
        out = {
            "hit@10": 100 * float(self.hit.mean()),
            "recall@10": 100 * float(self.recall.mean()),
            "precision@10": 100 * float(self.precision.mean()),
        }
        if reachable is not None:
            out["hit@10_reachable"] = 100 * float(self.hit[reachable].mean())
        return out


def per_customer_metrics(recs: np.ndarray, ev: EvalSet, n: int = TOP_N) -> PerCustomer:
    assert recs.shape == (ev.n_eval, n)
    r = np.repeat(np.arange(ev.n_eval), n)
    is_hit = np.asarray(ev.T[r, recs.ravel()]).reshape(ev.n_eval, n)
    hits = is_hit.sum(axis=1).astype(np.int64)
    return PerCustomer(
        hits=hits,
        hit=(hits > 0).astype(np.float64),
        recall=hits / ev.n_targets,
        precision=hits / n,
    )


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #
def config_name(k: int, weighting: str, row_source: str) -> str:
    return f"SVD k={k} {weighting} {row_source}"


def scoring_recommender(rec: Recommender, A_binary: sp.csr_matrix, row_source: str) -> Recommender:
    """Same factors, but customer rows for scoring taken from A_log ('log') or A_binary ('binary').

    A_binary has the identical sparsity pattern, so the already-bought mask is unchanged.
    """
    if row_source == "log":
        return rec
    if row_source == "binary":
        assert A_binary.shape == rec.A.shape and A_binary.nnz == rec.A.nnz
        return replace(rec, A=sp.csr_matrix(A_binary))
    raise ValueError(row_source)


def run_grid(
    A_log: sp.csr_matrix,
    A_binary: sp.csr_matrix,
    product_index: pd.DataFrame,
    ev: EvalSet,
    ks=K_GRID,
    weightings=WEIGHTINGS,
    row_sources=ROW_SOURCES,
    n: int = TOP_N,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, PerCustomer], dict[str, np.ndarray], dict[int, Recommender]]:
    """Fit one SVD per k on A_log; evaluate every (k, weighting, row_source)."""
    rows_out, per_cust, all_recs, base_recs = [], {}, {}, {}
    for k in ks:
        rec_k = fit_recommender(A_log, product_index, k=k, weighting="V", seed=SEED, n_iter=N_ITER)
        base_recs[k] = rec_k
        for w in weightings:
            rec_w = rec_k if w == "V" else with_weighting(rec_k, w)
            for src in row_sources:
                rec_s = scoring_recommender(rec_w, A_binary, src)
                recs, _, score_s = recommend_for_customers(rec_s, ev.rows, n=n)
                pc = per_customer_metrics(recs, ev, n)
                name = config_name(k, w, src)
                per_cust[name] = pc
                all_recs[name] = recs
                row = {"config": name, "k": k, "weighting": w, "row": src, **pc.summary(ev.reachable),
                       "fit_s": round(rec_k.fit_seconds, 3), "score_eval_s": round(score_s, 3)}
                rows_out.append(row)
                if verbose:
                    print(f"{name:<24} hit@10 {row['hit@10']:6.2f}%  recall@10 {row['recall@10']:6.2f}%  "
                          f"precision@10 {row['precision@10']:6.2f}%  (fit {row['fit_s']:.3f}s)", flush=True)
    return pd.DataFrame(rows_out), per_cust, all_recs, base_recs


def evaluate_baseline(rec: Recommender, ev: EvalSet, n: int = TOP_N) -> tuple[dict, PerCustomer, np.ndarray]:
    recs, secs = best_seller_recs(rec, ev.rows, n=n)
    pc = per_customer_metrics(recs, ev, n)
    row = {"config": "best-seller baseline", "k": 0, "weighting": "-", "row": "-", **pc.summary(ev.reachable),
           "fit_s": 0.0, "score_eval_s": round(secs, 3)}
    return row, pc, recs


# --------------------------------------------------------------------------- #
# Statistics and breakdowns
# --------------------------------------------------------------------------- #
def paired_bootstrap(a: PerCustomer, b: PerCustomer, n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Paired bootstrap over customers of mean(a - b) for hit@10 and recall@10 (percentage points)."""
    rng = np.random.default_rng(seed)
    n = len(a.hit)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = []
    for metric, da in (("hit@10", a.hit - b.hit), ("recall@10", a.recall - b.recall)):
        boots = 100 * da[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({
            "metric": metric,
            "mean_diff_pp": 100 * float(da.mean()),
            "ci95_low_pp": float(lo),
            "ci95_high_pp": float(hi),
            "share_resamples_diff_gt_0": float(np.mean(boots > 0)),
            "n_customers": n,
            "n_boot": n_boot,
        })
    return pd.DataFrame(out)


def history_sizes(A_log: sp.csr_matrix, ev: EvalSet) -> np.ndarray:
    """Distinct products bought in train per evaluation customer."""
    return np.diff(A_log.indptr)[ev.rows]


def bucket_labels(sizes: np.ndarray, buckets=HISTORY_BUCKETS) -> np.ndarray:
    labels = np.empty(len(sizes), dtype=object)
    for lo, hi in buckets:
        name = f"{lo}-{int(hi)}" if np.isfinite(hi) else f"{lo}+"
        labels[(sizes >= lo) & (sizes <= hi)] = name
    assert not any(l is None for l in labels)
    return labels


def breakdown_by_history(sizes: np.ndarray, named: dict[str, PerCustomer], buckets=HISTORY_BUCKETS) -> pd.DataFrame:
    labels = bucket_labels(sizes, buckets)
    order = [f"{lo}-{int(hi)}" if np.isfinite(hi) else f"{lo}+" for lo, hi in buckets]
    rows = []
    for name, pc in named.items():
        for b in order:
            m = labels == b
            rows.append({
                "config": name, "history_bucket": b, "n_customers": int(m.sum()),
                "hit@10": 100 * float(pc.hit[m].mean()) if m.any() else np.nan,
                "recall@10": 100 * float(pc.recall[m].mean()) if m.any() else np.nan,
            })
    return pd.DataFrame(rows)


def rare_product_hits(recs: np.ndarray, ev: EvalSet, product_index: pd.DataFrame, band=RARE_BAND, n: int = TOP_N) -> dict:
    """How many of a config's hits (top-n ∩ targets pairs) are products with n_customers in `band`."""
    r = np.repeat(np.arange(ev.n_eval), n)
    c = recs.ravel()
    is_hit = np.asarray(ev.T[r, c]).ravel().astype(bool)
    pop = product_index["n_customers"].to_numpy()
    hit_pop = pop[c[is_hit]]
    rec_pop = pop[c]
    in_band = (hit_pop >= band[0]) & (hit_pop <= band[1])
    # share of all in-matrix target pairs that fall in the band, for reference
    tgt_pop = pop[ev.T.indices]
    tgt_in_band = (tgt_pop >= band[0]) & (tgt_pop <= band[1])
    return {
        "total_hits": int(is_hit.sum()),
        f"hits_on_products_with_{band[0]}-{band[1]}_customers": int(in_band.sum()),
        "share_of_hits_pct": 100 * float(in_band.mean()) if is_hit.any() else np.nan,
        "share_of_recommendations_in_band_pct": 100 * float(((rec_pop >= band[0]) & (rec_pop <= band[1])).mean()),
        "share_of_in_matrix_targets_in_band_pct": 100 * float(tgt_in_band.mean()),
    }


# --------------------------------------------------------------------------- #
# Timing of the nightly job for one configuration
# --------------------------------------------------------------------------- #
def time_nightly(A_log: sp.csr_matrix, A_binary: sp.csr_matrix, product_index: pd.DataFrame,
                 k: int, weighting: str, row_source: str, repeats: int = 3, n: int = TOP_N) -> pd.DataFrame:
    """fit + item-neighbour precompute + score ALL customers; `repeats` runs, wall clock per stage."""
    rows = []
    all_rows = np.arange(A_log.shape[0])
    for i in range(repeats):
        t0 = time.perf_counter()
        rec = fit_recommender(A_log, product_index, k=k, weighting=weighting, seed=SEED, n_iter=N_ITER)
        t_fit = time.perf_counter() - t0
        _, t_nb = precompute_item_neighbors(rec, n=n)
        rec_s = scoring_recommender(rec, A_binary, row_source)
        _, _, t_score = recommend_for_customers(rec_s, all_rows, n=n)
        rows.append({"repeat": i + 1, "fit_s": t_fit, "neighbors_s": t_nb, "score_all_s": t_score,
                     "total_s": t_fit + t_nb + t_score})
    return pd.DataFrame(rows)
