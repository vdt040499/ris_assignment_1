# Multimodal Semantic Search over MS-COCO

Search 5,000 COCO val2017 images by free-form text or by an image. FastAPI +
Qdrant backend, React frontend. 5 embedding spaces + 2 baselines, with real
ablation numbers in `results/` (not estimates).

See `docs/report.md` for the full write-up, and `results/qualitative.md` for
14 good/bad examples with figures.

## Requirements

- Python 3.11, Node 18+
- Docker Desktop running (Qdrant)
- About 15GB free disk on Windows with Docker Desktop (COCO images + model
  checkpoints + Docker's own WSL2 virtual disk growth), less on Linux/macOS
  without that overhead. No GPU required.

## Running from scratch

```bash
docker compose up -d qdrant            # requires Docker Desktop running
pip install -r requirements.txt
cp .env.example .env

python tasks.py ingest                 # downloads COCO, builds corpus.jsonl + thumbnails
python tasks.py build --space all      # encode + load into Qdrant, ~30-35 min on CPU
python tasks.py querysets              # generates data/queryset_short.json (the other
                                        # query sets, queryset_vi.json / queryset_i2i.json,
                                        # are already committed — see the note below)
python tasks.py eval --axis all        # generates results/*.csv + *.md

python tasks.py serve                  # API at http://localhost:8000
```

In another terminal:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev                            # UI at http://localhost:5173
```

**Note on query sets.** `data/queryset_vi.json` (200 captions hand-translated
into Vietnamese, used for ablation axis 5) and `data/queryset_i2i.json` (200
randomly sampled image IDs, used for image-to-image evaluation) are **already
committed to the repo**, because the Vietnamese translation needs manual
review and can't be regenerated identically on every run. `python tasks.py
querysets` generates (or — if it already exists — skips, unless `--force` is
passed) `data/queryset_short.json`, the short keyword-style query set used for
ablation axis 3(b).

**Idempotent.** `python tasks.py build --space <name>` skips a space that's
already built; add `--force` to rebuild it. You can build spaces one at a
time, e.g. `python tasks.py build --space clip-b32`, instead of `--space all`
in one go — useful on a slower machine or to build overnight.

## Tests

```bash
python -m pytest                       # fast suite, no model downloads, no Qdrant needed
python -m pytest -m integration        # tests that load real models (verifies mclip-b32
                                        # shares its space with clip-b32, spec §5.3)
cd frontend && npm test && npm run typecheck
```

## Documentation

- Design: `docs/superpowers/specs/2026-09-23-multimodal-search-design.md`
- Report: `docs/report.md`
- Ablation result tables: `results/axis*.md` (with matching `.csv`), build
  metadata: `results/index_meta/*.json` (`data/index_meta/*.json` is
  regenerated on a fresh build and is gitignored, so it won't exist on a new
  checkout — use the committed copy under `results/` instead)
- Good/bad analysis (14 examples with figures): `results/qualitative.md`,
  images under `results/figures/`
- Demo recording script (not yet recorded — see the note in the file itself):
  `docs/demo-script.md`

## Troubleshooting

- **`Could not connect to Qdrant`** when running `build`/`eval`/`serve`: run
  `docker compose up -d qdrant`, then check `curl http://localhost:6333/readyz`.
- **`Space index not built` (409)** on `/search/text` or `/search/image`: run
  the exact command the API response names, e.g.
  `python tasks.py build --space siglip-b16`.
- **Build is much slower than estimated**: build spaces one at a time
  (`--space <name>` instead of `--space all`) so an interruption doesn't force
  a full restart — the command is idempotent, so re-running it is safe.
