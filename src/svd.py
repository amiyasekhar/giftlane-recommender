"""ML2 - Truncated SVD on the frozen A_log matrix: k sweep, error formula, scree plots.

Usage (from the repo root)::

    from src.svd import load_inputs, total_energy, sweep_k, validate_error_formula, ...

Nothing here reconstructs A_k densely for the real matrix. The relative reconstruction
error is computed from the singular values only:

    ||A - A_k||_F^2 = ||A||_F^2 - sum_{i<=k} sigma_i^2
    rel_error(k)    = sqrt(1 - sum_{i<=k} sigma_i^2 / ||A||_F^2)

which holds exactly for the fitted ``TruncatedSVD`` because its ``components_`` are
orthonormal and ``singular_values_[i] = ||A v_i||``. ``validate_error_formula`` checks
this numerically on small matrices where the dense reconstruction is cheap.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

SEED = 42
N_ITER = 7
K_GRID = (10, 20, 50, 100, 200)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_inputs(root: str | Path) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load the frozen A_log plus product/customer index maps (read-only)."""
    root = Path(root)
    processed = root / "data" / "processed"
    A = sp.load_npz(processed / "A_log.npz").tocsr()
    product_index = pd.read_parquet(processed / "product_index.parquet")
    customer_index = pd.read_parquet(processed / "customer_index.parquet")
    assert A.shape == (len(customer_index), len(product_index)), "index maps do not match A_log"
    return A, product_index, customer_index


# --------------------------------------------------------------------------- #
# Energy / error helpers
# --------------------------------------------------------------------------- #
def total_energy(A: sp.spmatrix) -> float:
    """||A||_F^2 computed from the stored non-zeros only."""
    return float(np.sum(A.data.astype(np.float64) ** 2))


def energy_kept(singular_values: np.ndarray, energy: float) -> float:
    return float(np.sum(np.asarray(singular_values, dtype=np.float64) ** 2) / energy)


def rel_error(singular_values: np.ndarray, energy: float) -> float:
    """sqrt(1 - sum(sigma^2)/||A||_F^2); clipped at 0 to absorb floating-point noise."""
    return float(np.sqrt(max(0.0, 1.0 - energy_kept(singular_values, energy))))


def factor_size_mb(n_customers: int, n_products: int, k: int) -> float:
    """float64 storage for U_k (n x k) and V_k (m x k)."""
    return (n_customers + n_products) * k * 8 / 1e6


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_svd(A: sp.spmatrix, k: int, seed: int = SEED, n_iter: int = N_ITER) -> tuple[TruncatedSVD, float]:
    model = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=n_iter, random_state=seed)
    t0 = time.perf_counter()
    model.fit(A)
    return model, time.perf_counter() - t0


def sweep_k(
    A: sp.spmatrix, ks: tuple[int, ...] = K_GRID, seed: int = SEED, n_iter: int = N_ITER, verbose: bool = True
) -> tuple[pd.DataFrame, dict[int, TruncatedSVD]]:
    """Fit one TruncatedSVD per k and tabulate time / error / energy / factor size."""
    energy = total_energy(A)
    n, m = A.shape
    rows, models = [], {}
    for k in ks:
        model, secs = fit_svd(A, k, seed=seed, n_iter=n_iter)
        s = model.singular_values_
        rows.append(
            {
                "k": k,
                "fit_s": round(secs, 3),
                "rel_error": rel_error(s, energy),
                "energy_kept_pct": 100 * energy_kept(s, energy),
                "factors_MB": factor_size_mb(n, m, k),
            }
        )
        models[k] = model
        if verbose:
            r = rows[-1]
            print(
                f"k={k:>4}: fit {r['fit_s']:6.2f}s | rel_error {r['rel_error']:.4f} | "
                f"energy kept {r['energy_kept_pct']:6.2f}% | factors {r['factors_MB']:.2f} MB",
                flush=True,
            )
    return pd.DataFrame(rows), models


# --------------------------------------------------------------------------- #
# Validation of the error formula (small matrices only)
# --------------------------------------------------------------------------- #
def validate_error_formula(seed: int = SEED, tol: float = 1e-6) -> dict:
    """Check the singular-value error formula against explicit dense reconstruction.

    1. random 200x200 sparse matrix, density 0.05, k=20: formula vs ||A - U S V^T||_F / ||A||_F
    2. [[3,0],[4,5]] with full np.linalg.svd: sigma = (3*sqrt(5), sqrt(5)); k=1 rel error = sqrt(0.1)
    """
    out: dict = {}

    # --- 1. random sparse 200 x 200, k = 20 ----------------------------------
    rng = np.random.default_rng(seed)
    R = sp.random(200, 200, density=0.05, format="csr", random_state=rng, dtype=np.float64)
    Rd = R.toarray()
    energy = total_energy(R)
    k = 20

    # 1a. exact SVD: formula must match the explicit dense reconstruction to machine precision.
    U, s, Vt = np.linalg.svd(Rd, full_matrices=False)
    A_k = (U[:, :k] * s[:k]) @ Vt[:k]
    explicit = float(np.linalg.norm(Rd - A_k, "fro") / np.sqrt(energy))
    formula = rel_error(s[:k], energy)
    assert abs(explicit - formula) < tol, f"random 200x200 k=20: explicit {explicit:.10f} vs formula {formula:.10f}"

    # 1b. the same with sklearn's randomized TruncatedSVD. Its U*Sigma comes from a random sketch,
    #     so U S V^T is not exactly the orthogonal projection A V^T V and the two numbers differ by
    #     the sketch gap. Reported, and asserted to be small, but this is NOT a test of the formula.
    model = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=N_ITER, random_state=seed)
    US = model.fit_transform(R)
    explicit_rand = float(np.linalg.norm(Rd - US @ model.components_, "fro") / np.sqrt(energy))
    formula_rand = rel_error(model.singular_values_, energy)
    sigma_gap = float(np.max(np.abs(model.singular_values_ - s[:k])))
    assert abs(explicit_rand - formula_rand) < 1e-2

    out["random_200x200"] = {
        "nnz": int(R.nnz),
        "k": k,
        "exact_svd": {
            "rel_error_formula": formula,
            "rel_error_explicit": explicit,
            "abs_diff": abs(explicit - formula),
        },
        "randomized_truncated_svd": {
            "rel_error_formula": formula_rand,
            "rel_error_explicit": explicit_rand,
            "abs_diff": abs(explicit_rand - formula_rand),
            "max_abs_gap_in_singular_values_vs_exact": sigma_gap,
        },
    }

    # --- 2. tiny exact case --------------------------------------------------
    M = np.array([[3.0, 0.0], [4.0, 5.0]])
    _, s, _ = np.linalg.svd(M)
    expected_s = np.array([3 * np.sqrt(5), np.sqrt(5)])
    assert np.allclose(s, expected_s, atol=1e-12), f"singular values {s} != {expected_s}"
    e2 = float(np.sum(M**2))  # = 50
    k1 = rel_error(s[:1], e2)
    assert abs(k1 - np.sqrt(0.1)) < 1e-12, f"k=1 rel error {k1} != sqrt(0.1)"
    # cross-check with explicit rank-1 reconstruction
    U, s_, Vt = np.linalg.svd(M)
    M1 = s_[0] * np.outer(U[:, 0], Vt[0])
    k1_explicit = float(np.linalg.norm(M - M1, "fro") / np.linalg.norm(M, "fro"))
    assert abs(k1_explicit - np.sqrt(0.1)) < 1e-12
    out["matrix_2x2"] = {
        "singular_values": s.tolist(),
        "expected": expected_s.tolist(),
        "frobenius_sq": e2,
        "k1_rel_error_formula": k1,
        "k1_rel_error_explicit": k1_explicit,
        "expected_k1_rel_error": float(np.sqrt(0.1)),
    }
    return out


def projection_rel_error(A: sp.spmatrix, model: TruncatedSVD, energy: float | None = None) -> float:
    """Exact relative error of the model's own reconstruction A_k = A V^T V, without forming A_k.

    Because the rows of ``components_`` (V) are orthonormal,
        ||A - A V^T V||_F^2 = ||A||_F^2 - ||A V^T||_F^2,
    and ``A V^T`` is only n x k, so this costs one sparse matmul. On the real matrix it should
    agree with ``rel_error(model.singular_values_, energy)`` up to the randomized-sketch gap.
    """
    energy = total_energy(A) if energy is None else energy
    AVt = A @ model.components_.T  # n x k, dense but small
    kept = float(np.sum(AVt**2)) / energy
    return float(np.sqrt(max(0.0, 1.0 - kept)))


# --------------------------------------------------------------------------- #
# Interpretation helpers
# --------------------------------------------------------------------------- #
def top_products_for_component(
    model: TruncatedSVD, product_index: pd.DataFrame, component: int, n: int = 10
) -> pd.DataFrame:
    """Top-n products by |weight| in ``components_[component]`` (0-based), with sign kept."""
    v = model.components_[component]
    idx = np.argsort(-np.abs(v))[:n]
    out = pd.DataFrame(
        {
            "StockCode": product_index["StockCode"].to_numpy()[idx],
            "Description": product_index["Description"].to_numpy()[idx],
            "weight": v[idx],
            "n_customers_train": product_index["n_customers"].to_numpy()[idx],
        }
    )
    out.index = range(1, len(out) + 1)
    out.index.name = "rank"
    return out


def error_curve(singular_values: np.ndarray, energy: float) -> pd.DataFrame:
    """Relative error and energy kept for every k = 1..len(sigma) from one fit's singular values."""
    s2 = np.cumsum(np.asarray(singular_values, dtype=np.float64) ** 2)
    kept = s2 / energy
    return pd.DataFrame(
        {
            "k": np.arange(1, len(s2) + 1),
            "energy_kept_pct": 100 * kept,
            "rel_error": np.sqrt(np.clip(1 - kept, 0, None)),
        }
    )
