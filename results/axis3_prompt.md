### Trục 3 — prompt template trên caption dài và trên query ngắn

| space | n_queries | exact | hnsw_ef | prompt_template | R@1 | R@5 | R@10 | MRR@10 | search_ms_p50 | search_ms_p95 | P@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 | 5000 | True | None |  | 0.3138 | 0.5536 | 0.659 | 0.4152 | 3.382 | 49.255 |  |
| clip-b32 | 80 |  |  |  |  |  |  |  | 3.435 |  | 0.8012 |
| clip-b32 | 5000 | True | None | a photo of {} | 0.309 | 0.5528 | 0.6626 | 0.4145 | 3.425 | 49.365 |  |
| clip-b32 | 80 |  |  | a photo of {} |  |  |  |  | 3.416 |  | 0.7962 |
| clip-b32 | 5000 | True | None | a photo of {}, a type of scene | 0.2866 | 0.5304 | 0.6438 | 0.3905 | 3.434 | 48.993 |  |
| clip-b32 | 80 |  |  | a photo of {}, a type of scene |  |  |  |  | 3.571 |  | 0.7812 |
