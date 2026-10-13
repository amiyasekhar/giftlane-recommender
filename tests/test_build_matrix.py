"""Hand-worked toy example for ``src.build_matrix.build_matrix``.

Transactions (customer, product, quantity):
    (C1, Mug, 12), (C1, Mug, 12), (C1, Candle, 6),
    (C2, Bunting, 48),
    (C3, Mug, 6), (C3, Bunting, 6),
    (C4, Candle, 24)

Products sort to Bunting, Candle, Mug -> columns 0, 1, 2.  Customers sort to C1..C4 -> rows 0..3.
"""

import numpy as np
import pandas as pd
import pytest

from src.build_matrix import build_matrix

TOY = pd.DataFrame(
    {
        "Customer ID": ["C1", "C1", "C1", "C2", "C3", "C3", "C4"],
        "StockCode": ["Mug", "Mug", "Candle", "Bunting", "Mug", "Bunting", "Candle"],
        "Quantity": [12, 12, 6, 48, 6, 6, 24],
    }
)

# Column order after sorting: Bunting, Candle, Mug
EXPECTED_RAW = np.array(
    [
        [0, 6, 24],  # C1: Candle 6, Mug 12+12
        [48, 0, 0],  # C2: Bunting 48
        [6, 0, 6],  # C3: Bunting 6, Mug 6
        [0, 24, 0],  # C4: Candle 24
    ],
    dtype=np.float64,
)


@pytest.fixture(scope="module")
def result():
    return build_matrix(TOY, value="log", min_customers=1)


def test_index_maps_are_sorted(result):
    assert result.customer_index["Customer ID"].tolist() == ["C1", "C2", "C3", "C4"]
    assert result.customer_index["row"].tolist() == [0, 1, 2, 3]
    assert result.product_index["StockCode"].tolist() == ["Bunting", "Candle", "Mug"]
    assert result.product_index["col"].tolist() == [0, 1, 2]
    assert result.product_index["n_customers"].tolist() == [2, 2, 2]


def test_raw_matrix_sums_quantity_per_pair(result):
    np.testing.assert_array_equal(result.matrices["raw"].toarray(), EXPECTED_RAW)


def test_log_matrix_is_log1p_of_aggregated_raw(result):
    A_log = result.matrices["log"].toarray()
    np.testing.assert_allclose(A_log, np.log1p(EXPECTED_RAW))
    # The whole point: C1 bought 12 + 12 Mugs -> log(1 + 24) = log(25), NOT log(13) + log(13) = log(169).
    c1, mug = 0, 2
    assert A_log[c1, mug] == pytest.approx(np.log(25))
    assert A_log[c1, mug] != pytest.approx(np.log(169))


def test_binary_matrix_has_six_ones(result):
    A_bin = result.matrices["binary"]
    assert A_bin.nnz == 6
    assert np.all(A_bin.data == 1.0)
    assert A_bin.sum() == 6


def test_nnz_equals_number_of_pairs(result):
    for A in result.matrices.values():
        assert A.nnz == 6
        assert A.shape == (4, 3)
        assert A.dtype == np.float64
    assert len(result.pairs) == 6


def test_matrices_share_sparsity_pattern(result):
    ref = result.matrices["log"]
    for A in result.matrices.values():
        np.testing.assert_array_equal(A.indptr, ref.indptr)
        np.testing.assert_array_equal(A.indices, ref.indices)


def test_main_matrix_follows_value_argument():
    r = build_matrix(TOY, value="binary", min_customers=1)
    assert r.A is r.matrices["binary"]


def test_min_customers_filter_drops_products_and_empty_customers():
    # Add a product bought by a single customer, and a customer who only bought that product.
    toy = pd.concat(
        [TOY, pd.DataFrame({"Customer ID": ["C5"], "StockCode": ["Rare"], "Quantity": [1]})],
        ignore_index=True,
    )
    r = build_matrix(toy, value="log", min_customers=2)
    assert r.product_index["StockCode"].tolist() == ["Bunting", "Candle", "Mug"]
    assert r.customer_index["Customer ID"].tolist() == ["C1", "C2", "C3", "C4"]
    assert r.stats["products_dropped"] == 1
    assert r.stats["customers_dropped"] == 1
    assert r.A.nnz == 6
