### Trục 5 — cùng 200 nội dung, tiếng Anh vs tiếng Việt

| space | n_queries | exact | hnsw_ef | prompt_template | R@1 | R@5 | R@10 | MRR@10 | search_ms_p50 | search_ms_p95 | language |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 | 200 | True | None |  | 0.255 | 0.51 | 0.67 | 0.3663 | 3.487 | 49.793 | en |
| clip-b32 | 200 | True | None |  | 0.0 | 0.025 | 0.03 | 0.0093 | 3.399 | 4.12 | vi |
| mclip-b32 | 200 | True | None |  | 0.195 | 0.435 | 0.62 | 0.3045 | 3.386 | 47.05 | en |
| mclip-b32 | 200 | True | None |  | 0.13 | 0.34 | 0.445 | 0.2134 | 3.48 | 45.924 | vi |
