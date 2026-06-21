# Mean DNABERT-2 Embedding Exploration

- HDF5 files discovered: 172500
- Embeddings analyzed: 2000
- Analysis level: bag
- Metadata matches: 1954/2000
- PCA explained variance: PC1=0.9751, PC2=0.0112

## Column Notes

- `Elevation` is site altitude relative to sea level and is populated mainly for NEON soil samples.
- `Depth_m` is sampling depth: soil core depth for soil samples and water-column depth for marine samples.
- `embedding_variance` is the per-DNABERT-coordinate variance across the reads in that one bag.
- `read_spread_l2 = sqrt(sum(embedding_variance))` is the typical read-level spread around the bag mean in DNABERT space.
- `sem_l2_estimate = sqrt(sum(embedding_variance) / num_reads)` estimates uncertainty of the bag mean, not ecological heterogeneity by itself.
- `spread_to_nn_ratio` compares read-level spread to the distance from this bag mean to its nearest other bag mean; large values flag means that may average over a broad/multimodal read cloud.
