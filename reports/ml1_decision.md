# ML1 decision note: Truncated SVD vs PCA

## Decision

Truncated SVD (scikit-learn `TruncatedSVD`) is the main model. PCA is run on the same matrix with the same k only as the ML5 comparison.

## Why

- Purchase data is implicit feedback: it records what customers bought, never what they disliked. A zero usually means the customer never saw the product, not that they rejected it.
- PCA centres each column (subtracts its mean). Every zero becomes negative, so a non-purchase reads as "below average", close to a dislike, which the data cannot support. SVD keeps a zero meaning "no signal".
- PCA describes how each customer differs from the average customer. SVD describes what customers bought together, which is what "Customers who bought this also bought" needs.
- Memory: centring destroys sparsity. A_log is 97.95% zeros: 5.0 MB sparse, 161.4 MB dense once centred. TruncatedSVD works directly on the sparse matrix.

## Expected trade-off

SVD's component 1 likely captures overall purchase volume rather than taste, so the customer map (ML6) may need components 2 and 3, or row-normalised scores. Checked in ML5/ML6.

## Hand-check

On a toy 4x3 binary matrix (each product bought by 2 of 4 customers), centring by the column mean 0.5 made all 12 cells non-zero: sparsity fell from 50% to 0%.
