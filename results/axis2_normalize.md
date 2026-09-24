### Trục 2 — cosine trên vector normalize vs dot trên vector thô

| space | n_queries | exact | hnsw_ef | prompt_template | R@1 | R@5 | R@10 | MRR@10 | search_ms_p50 | search_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 | 5000 | True | None |  | 0.3138 | 0.5536 | 0.659 | 0.4152 | 3.357 | 49.089 |
| clip-b32-raw | 5000 | True | None |  | 0.2636 | 0.493 | 0.603 | 0.3618 | 3.353 | 48.159 |
