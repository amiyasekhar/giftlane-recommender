"""Tiny hand-built matrix for ``src.recommend``.

Products 0 and 1 are always bought together, products 2 and 3 are always bought together,
and neither pair is ever bought with the other. Product 4 is never bought by anyone
(an all-zero column), to check zero-norm handling.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.recommend import (
    best_seller_recs,
    fit_recommender,
    precompute_item_neighbors,
    product_vectors,
    recommend_for_customers,
    similar_products,
    with_weighting,
)

#            p0 p1 p2 p3 p4
A_DENSE = np.array(
    [
        [1, 1, 0, 0, 0],  # row 0
        [1, 1, 0, 0, 0],  # row 1
        [1, 1, 0, 0, 0],  # row 2
        [0, 0, 1, 1, 0],  # row 3
        [0, 0, 1, 1, 0],  # row 4
        [0, 0, 1, 1, 0],  # row 5
        [1, 0, 0, 0, 0],  # row 6: only product 0
        [0, 0, 1, 0, 0],  # row 7: only product 2
    ],
    dtype=np.float64,
)
A = sp.csr_matrix(A_DENSE)
PRODUCT_INDEX = pd.DataFrame(
    {
        "StockCode": ["P0", "P1", "P2", "P3", "P4"],
        "col": np.arange(5),
        "Description": ["mug", "mug lid", "candle", "candle holder", "never bought"],
        "n_customers": (A_DENSE > 0).sum(axis=0).astype(int),
    }
)


@pytest.fixture(scope="module", params=["V", "VSigma"])
def rec(request):
    return fit_recommender(A, PRODUCT_INDEX, k=2, weighting=request.param)


def test_product_vectors_are_unit_norm_and_zero_rows_are_safe(rec):
    norms = np.linalg.norm(rec.P, axis=1)
    assert np.allclose(norms[:4], 1.0)
    assert norms[4] == 0.0  # all-zero product column -> zero vector, no NaN
    assert not np.isnan(rec.P).any()
    assert rec.zero_norm_cols.tolist() == [4]


def test_product_1_is_top_neighbour_of_product_0(rec):
    idx, scores = similar_products(rec, 0, n=4)
    assert idx[0] == 1
    assert scores[0] == pytest.approx(1.0, abs=1e-9)
    assert 0 not in idx  # never returns itself
    # the other pair (and the never-bought product) are orthogonal to product 0
    sim = dict(zip(idx.tolist(), scores.tolist()))
    assert sim[2] == pytest.approx(0.0, abs=1e-9)
    assert sim[3] == pytest.approx(0.0, abs=1e-9)
    assert sim[4] == pytest.approx(0.0, abs=1e-9)


def test_similar_products_never_returns_self(rec):
    for j in range(rec.n_products):
        idx, _ = similar_products(rec, j, n=rec.n_products - 1)
        assert j not in idx


def test_item_neighbors_table(rec):
    table, secs = precompute_item_neighbors(rec, n=2, batch_size=2)
    assert len(table) == rec.n_products * 2
    assert secs >= 0
    top_of_0 = table.loc[(table["product_col"] == 0) & (table["rank"] == 1), "neighbor_col"].iloc[0]
    top_of_2 = table.loc[(table["product_col"] == 2) & (table["rank"] == 1), "neighbor_col"].iloc[0]
    assert top_of_0 == 1 and top_of_2 == 3
    assert not (table["product_col"] == table["neighbor_col"]).any()


def test_customer_recs_exclude_already_bought(rec):
    rows = np.arange(rec.n_customers)
    recs, scores, _ = recommend_for_customers(rec, rows, n=2, batch_size=3)
    for r, cols in zip(rows, recs):
        bought = set(A[r].indices)
        assert bought.isdisjoint(cols), f"row {r}: recommended already-bought {bought & set(cols)}"
    # row 6 bought only product 0 -> product 1 should be recommended first
    assert recs[6][0] == 1
    # row 7 bought only product 2 -> product 3 first
    assert recs[7][0] == 3


def test_best_seller_baseline_excludes_bought(rec):
    rows = np.arange(rec.n_customers)
    recs, _ = best_seller_recs(rec, rows, n=2)
    for r, cols in zip(rows, recs):
        assert set(A[r].indices).isdisjoint(cols)
    # products 0 and 2 have 4 customers each (the most); row 3 bought 2,3 so gets 0 then 1
    assert recs[3].tolist() == [0, 1]


def test_weightings_share_the_svd_and_differ_only_in_scaling():
    r_v = fit_recommender(A, PRODUCT_INDEX, k=2, weighting="V")
    r_vs = with_weighting(r_v, "VSigma")
    assert r_vs.svd is r_v.svd
    P_v, _ = product_vectors(r_v.svd, "V")
    P_vs, _ = product_vectors(r_v.svd, "VSigma")
    assert P_v.shape == P_vs.shape == (5, 2)
    with pytest.raises(ValueError):
        product_vectors(r_v.svd, "bogus")
