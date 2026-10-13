"""ML6 - customer map: SVD customer vectors, volume removal, k-means segments, profiles, 2-D maps.

Usage (from the repo root)::

    from src.segments import customer_vectors, normalise_rows, volume_correlations, cluster_sweep, ...

Customer vectors are ``U * Sigma = svd.transform(A_log)`` (n x k). Component 1 is overall buying volume
(ML2 / ML5), so every vector is scaled to unit L2 length before clustering; what remains is the *mix*
of what a customer buys, not how much.

Nothing here reads test data. Segment numbering is by size (Segment 1 = largest) so that tables and
figures agree. Customer IDs and row indices never appear in any output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.svd import SEED

N_INIT = 20
CLUSTER_GRID = (3, 4, 5, 6)
SILHOUETTE_SAMPLE = 3000
MIN_CLUSTER_SHARE = 0.10
N_SIGNATURE = 10
N_COUNTRIES = 3


# --------------------------------------------------------------------------- #
# Customer vectors
# --------------------------------------------------------------------------- #
def customer_vectors(svd: TruncatedSVD, A: sp.csr_matrix) -> np.ndarray:
    """U * Sigma, shape (n_customers, k)."""
    X = svd.transform(A)
    assert X.shape == (A.shape[0], svd.n_components)
    return X


def normalise_rows(X: np.ndarray) -> np.ndarray:
    """Unit-L2 rows (removes overall volume). Rows that are all zero stay zero."""
    return normalize(X, norm="l2", axis=1, copy=True)


def volume_correlations(X_raw: np.ndarray, X_unit: np.ndarray, A: sp.csr_matrix) -> pd.DataFrame:
    """Correlation of the component-1 score with the number of products bought, before/after normalising."""
    n_products = np.diff(A.indptr)
    rows = []
    for stage, X in (("raw U*Sigma", X_raw), ("unit-norm", X_unit)):
        c1 = X[:, 0]
        rows.append(
            {
                "vectors": stage,
                "pearson_r(component1, n_products)": pearsonr(c1, n_products)[0],
                "spearman_rho(component1, n_products)": spearmanr(c1, n_products)[0],
                "component1_min": float(c1.min()),
                "component1_max": float(c1.max()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def fit_kmeans(X: np.ndarray, n_clusters: int, seed: int = SEED, n_init: int = N_INIT) -> KMeans:
    return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit(X)


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Map raw k-means labels to 1..n with 1 = largest cluster."""
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.lexsort((uniq, -counts))  # largest first, ties by raw label
    mapping = {int(uniq[i]): new for new, i in enumerate(order, start=1)}
    return np.vectorize(mapping.get)(labels)


def silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int = SILHOUETTE_SAMPLE, seed: int = SEED) -> float:
    return float(silhouette_score(X, labels, sample_size=min(sample_size, len(labels)), random_state=seed))


def cluster_sweep(
    X_unit: np.ndarray, grid=CLUSTER_GRID, seed: int = SEED, n_init: int = N_INIT, verbose: bool = True
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Fit k-means for each n_clusters; return a summary table and size-ordered labels (1..n) per n."""
    rows, labels_by_n = [], {}
    for n in grid:
        km = fit_kmeans(X_unit, n, seed=seed, n_init=n_init)
        labels = relabel_by_size(km.labels_)
        sil = silhouette(X_unit, labels, seed=seed)
        sizes = pd.Series(labels).value_counts().sort_index()
        labels_by_n[n] = labels
        rows.append(
            {
                "n_clusters": n,
                "silhouette": sil,
                "inertia": float(km.inertia_),
                "sizes": " / ".join(f"{s:,}" for s in sizes.to_numpy()),
                "smallest_pct": 100 * sizes.min() / len(labels),
            }
        )
        if verbose:
            print(f"n_clusters={n}: silhouette {sil:.4f} | inertia {km.inertia_:.1f} | sizes {rows[-1]['sizes']}", flush=True)
    return pd.DataFrame(rows), labels_by_n


# --------------------------------------------------------------------------- #
# Per-customer statistics from the training data (no test data)
# --------------------------------------------------------------------------- #
def customer_stats(train: pd.DataFrame, customer_index: pd.DataFrame) -> pd.DataFrame:
    """One row per matrix row: products bought, total spend, invoices, main country (train only)."""
    t = train[["Customer ID", "StockCode", "Invoice", "Quantity", "Price", "Country"]].copy()
    t["spend"] = t["Quantity"] * t["Price"]
    g = t.groupby("Customer ID")
    stats = pd.DataFrame(
        {
            "n_products": g["StockCode"].nunique(),
            "spend": g["spend"].sum(),
            "n_invoices": g["Invoice"].nunique(),
            "country": g["Country"].agg(lambda s: s.value_counts().idxmax()),
        }
    )
    out = customer_index.merge(stats, left_on="Customer ID", right_index=True, how="left")
    assert out["n_products"].notna().all(), "a matrix customer has no train rows"
    out = out.sort_values("row").reset_index(drop=True)
    assert (out["row"].to_numpy() == np.arange(len(out))).all()
    return out.drop(columns=["Customer ID"])  # never carried further


# --------------------------------------------------------------------------- #
# Segment profiles
# --------------------------------------------------------------------------- #
def signature_products(
    A: sp.csr_matrix,
    labels: np.ndarray,
    product_index: pd.DataFrame,
    n_signature: int = N_SIGNATURE,
    min_cluster_share: float = MIN_CLUSTER_SHARE,
) -> pd.DataFrame:
    """Top products per segment by lift = P(bought | segment) / P(bought), among products bought by
    at least `min_cluster_share` of the segment's customers."""
    B = (A != 0).astype(np.float64).tocsr()
    overall = np.asarray(B.mean(axis=0)).ravel()
    rows = []
    for seg in np.unique(labels):
        m = labels == seg
        share = np.asarray(B[m].mean(axis=0)).ravel()
        lift = np.divide(share, overall, out=np.zeros_like(share), where=overall > 0)
        ok = share >= min_cluster_share
        cand = np.flatnonzero(ok)
        top = cand[np.lexsort((-share[cand], -lift[cand]))][:n_signature]
        for r, j in enumerate(top, start=1):
            rows.append(
                {
                    "segment": int(seg),
                    "rank": r,
                    "StockCode": product_index["StockCode"].iat[j],
                    "Description": product_index["Description"].iat[j],
                    "segment_share_pct": 100 * share[j],
                    "overall_share_pct": 100 * overall[j],
                    "lift": lift[j],
                    "n_eligible_products": int(ok.sum()),
                }
            )
    return pd.DataFrame(rows)


def segment_summary(labels: np.ndarray, stats: pd.DataFrame, n_countries: int = N_COUNTRIES) -> pd.DataFrame:
    rows = []
    n = len(labels)
    for seg in np.unique(labels):
        m = labels == seg
        s = stats.loc[m]
        ctry = s["country"].value_counts(normalize=True).head(n_countries)
        rows.append(
            {
                "segment": int(seg),
                "segment_name": "",
                "n_customers": int(m.sum()),
                "pct_customers": 100 * m.sum() / n,
                "median_products": float(s["n_products"].median()),
                "median_spend": float(s["spend"].median()),
                "median_invoices": float(s["n_invoices"].median()),
                "top_countries": "; ".join(f"{c} {100 * p:.1f}%" for c, p in ctry.items()),
            }
        )
    return pd.DataFrame(rows)


def segment_profiles(
    labels: np.ndarray, stats: pd.DataFrame, A: sp.csr_matrix, product_index: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(summary, signature, long) where `long` is the single-table form saved to CSV."""
    summary = segment_summary(labels, stats)
    sig = signature_products(A, labels, product_index)
    s_long = summary.assign(section="summary")
    g_long = sig.assign(section="signature_product")
    long = pd.concat([s_long, g_long], ignore_index=True, sort=False)
    cols = ["section", "segment", "segment_name", "n_customers", "pct_customers", "median_products", "median_spend",
            "median_invoices", "top_countries", "rank", "StockCode", "Description", "segment_share_pct",
            "overall_share_pct", "lift", "n_eligible_products"]
    return summary, sig, long[cols]


def save_profiles(long: pd.DataFrame, root: str | Path, n_clusters: int) -> Path:
    path = Path(root) / "reports" / f"ml6_segments_k{n_clusters}.csv"
    long.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# 2-D maps
# --------------------------------------------------------------------------- #
def project_2d(X_unit: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, TruncatedSVD]:
    """2-D TruncatedSVD of the unit-norm customer vectors (fitted on the 50-dim rows)."""
    p = TruncatedSVD(n_components=2, algorithm="randomized", n_iter=7, random_state=seed)
    Y = p.fit_transform(X_unit)
    return Y, p


def axis_words(
    direction: np.ndarray, product_index: pd.DataFrame, orientation: str = "x", n: int = 2, max_chars: int = 26
) -> str:
    """Plain-word label for an axis direction given in PRODUCT space (length m): the products with the
    largest positive and largest negative weight. 'x' -> one line (← low | high →); 'y' -> two lines (↑ / ↓)."""
    desc = product_index["Description"].str.strip().str.title().to_numpy()

    def short(s: str) -> str:
        return s if len(s) <= max_chars else s[: max_chars - 1] + "…"

    pos = np.argsort(-direction)[:n]
    neg = np.argsort(direction)[:n]
    low = ", ".join(short(desc[j]) for j in neg if direction[j] < 0) or "(no negative weights)"
    high = ", ".join(short(desc[j]) for j in pos if direction[j] > 0)
    if orientation == "x":
        return f"←  {low}    |    {high}  →"
    return f"↑ {high}\n↓ {low}"


def component_axis_words(svd: TruncatedSVD, component: int, product_index: pd.DataFrame, orientation: str = "x") -> str:
    return axis_words(svd.components_[component], product_index, orientation)


def projection_axis_words(
    proj: TruncatedSVD, svd: TruncatedSVD, axis: int, product_index: pd.DataFrame, orientation: str = "x"
) -> str:
    """A 2-D axis lives in the 50-dim component space; map it to product space via V (50 x m)."""
    direction = proj.components_[axis] @ svd.components_
    return axis_words(direction, product_index, orientation)


def centroids_2d(Y: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(s): Y[labels == s].mean(axis=0) for s in np.unique(labels)}


def spread_labels(points: dict[int, np.ndarray], Y: np.ndarray, min_frac: float = 0.07) -> dict[int, np.ndarray]:
    """Nudge label positions apart vertically when two centroids sit closer than `min_frac` of the
    axis ranges in both x and y, so text boxes do not overlap. The data points are not moved."""
    xr = np.ptp(Y[:, 0]) or 1.0
    yr = np.ptp(Y[:, 1]) or 1.0
    keys = sorted(points, key=lambda s: points[s][1])
    pos = {s: points[s].astype(float).copy() for s in keys}
    for _ in range(10):
        moved = False
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                dx = abs(pos[a][0] - pos[b][0]) / xr
                dy = abs(pos[a][1] - pos[b][1]) / yr
                if dx < min_frac * 2.2 and dy < min_frac:
                    shift = (min_frac - dy) / 2 * yr + 1e-9
                    lo, hi = (a, b) if pos[a][1] <= pos[b][1] else (b, a)
                    pos[lo][1] -= shift
                    pos[hi][1] += shift
                    moved = True
        if not moved:
            break
    return pos
