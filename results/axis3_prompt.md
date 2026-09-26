### Trục 3 — prompt template trên caption dài và trên query ngắn

| space | n_queries | exact | hnsw_ef | prompt_template | R@1 | R@5 | R@10 | MRR@10 | search_ms_p50 | search_ms_p95 | P@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 | 5000 | True | None |  | 0.3136 | 0.552 | 0.659 | 0.4152 | 1.996 | 3.038 |  |
| clip-b32 | 80 |  |  |  |  |  |  |  | 1.968 |  | 0.8012 |
| clip-b32 | 5000 | True | None | a photo of {} | 0.3106 | 0.5528 | 0.663 | 0.4153 | 2.45 | 3.758 |  |
| clip-b32 | 80 |  |  | a photo of {} |  |  |  |  | 2.239 |  | 0.7962 |
| clip-b32 | 5000 | True | None | a photo of {}, a type of scene | 0.2874 | 0.5298 | 0.6444 | 0.3912 | 2.79 | 3.693 |  |
| clip-b32 | 80 |  |  | a photo of {}, a type of scene |  |  |  |  | 2.081 |  | 0.78 |
