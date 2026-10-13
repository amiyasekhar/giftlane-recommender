# ML5 - PCA vs Truncated SVD (k = 50) on the frozen A_log matrix

| Measure | PCA | Truncated SVD |
|---|---|---|
| Estimator | PCA(n_components=50, svd_solver='arpack', random_state=42) | TruncatedSVD(n_components=50, algorithm='randomized', n_iter=7, random_state=42) |
| Input | same sparse A_log (5,471 x 3,687); columns centred implicitly by sklearn | sparse A_log as stored (no centring) |
| Solver actually used | arpack | randomized |
| Fit time (s), median of 3 | 0.096 | 0.101 |
| Peak memory during fit (MB, tracemalloc) | 8.3 | 9.7 |
| Stored factors (MB, float64: customer scores + product components) | 3.66 (+0.03 for mean_) | 3.66 |
| Relative reconstruction error sqrt(1 - sum(sigma^2)/||A||_F^2) | n/a (centred model) | 0.7892 |
| Energy kept (% of ||A||_F^2) | n/a | 37.71% |
| Explained variance (% of centred variance) | 34.99% | n/a |
| hit@10 (1,931 eval customers) | 20.66% | 21.44% |
| recall@10 | 2.26% | 2.35% |
| precision@10 | 3.09% | 3.15% |
| Best-seller baseline hit@10 (same for both) | 26.77% | 26.77% |
| Scoring note | customer rows NOT centred at scoring time (raw A_log rows @ P @ P.T) | raw A_log rows @ P @ P.T |
| Component 1: top 10 products by absolute weight | PACK OF 72 RETRO SPOT CAKE CASES; JUMBO BAG RED RETROSPOT; LUNCH BAG RED RETROSPOT; WHITE HANGING HEART T-LIGHT HOLDER; 60 TEATIME FAIRY CAKE CASES; LUNCH BAG SPACEBOY DESIGN; PACK OF 60 PINK PAISLEY CAKE CASES; LUNCH BAG PINK POLKADOT; LUNCH BAG CARS BLUE; LUNCH BAG  BLACK SKULL. | PACK OF 72 RETRO SPOT CAKE CASES; WHITE HANGING HEART T-LIGHT HOLDER; JUMBO BAG RED RETROSPOT; LUNCH BAG RED RETROSPOT; 60 TEATIME FAIRY CAKE CASES; ASSORTED COLOUR BIRD ORNAMENT; PACK OF 60 PINK PAISLEY CAKE CASES; LUNCH BAG SPACEBOY DESIGN; SET/20 RED SPOTTY PAPER NAPKINS; LUNCH BAG PINK POLKADOT |
| Component 1: sign pattern | mixed: 3,680 positive / 7 negative of 3,687 | all one sign (positive) |
| Component 1: share of variance / energy | 9.30% | 13.02% |
| cosine(component 1, normalised column means of A_log) | +0.9660 | +0.9856 |
| What component 1 represents, in words | Nearly the same direction (0.996 to SVD's). After centring it measures how much a customer differs from the average in volume, so heavy buyers sit at one end and light buyers at the other. Seven products have small negative weights. | Overall buying volume. It points almost exactly along the average customer's basket (cosine 0.99) and its top products are the best sellers. Every weight is positive, so customers who buy more of the popular range score higher. |

Notes:
- Fit time is the median of 3 fresh fits; peak memory is from one extra fit with `tracemalloc` (Python-allocator view; ARPACK/LAPACK workspaces are not counted).
- Both methods are scored with the identical pipeline from `src/recommend.py`: P = components_.T with unit-norm rows, scores = A_log[row] @ P @ P.T, already-bought products excluded, top 10. PCA rows are NOT centred at scoring time.
- Component signs are arbitrary; 'all one sign' is the pattern that matters, not positive vs negative.
- Source notebook: `notebooks/08_pca_vs_svd.ipynb`.

## Why PCA and SVD agree here

| Statistic of A_log | Value |
|---|---|
| Mean of all column means | 0.0436 |
| Max column mean | 0.7382 |
| Mean non-zero value | 2.1271 |
| Mean column mean as % of mean non-zero value | 2.05% |

Because A_log is 97.95% zeros, each column mean is small relative to the non-zero values, so centring shifts the data very little and PCA finds almost the same directions as SVD (component 1 cosine 0.996). The ML1 concern about centring is correct in principle but has a small effect on this matrix.
