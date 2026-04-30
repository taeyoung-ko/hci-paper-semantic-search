# HCI Paper Semantic Search

Two-stage semantic search over 13,488 papers from CHI, UIST, CSCW, DIS, IUI, RecSys, and UMAP.
- Embed (Qwen3-Embedding-0.6B) → FAISS top-K
- Rerank (Qwen3-Reranker-8B cross-encoder) → top-N
Split your query into research components (Background / Gap / Solution / Approach·Method / Findings). Each filled box runs as its own instruction-tuned search. With two or more boxes filled, a **Combined** tab fuses the rankings via RRF. Star papers to build a session collection, attach notes, and export to `.bib`.
Runs locally via Docker Compose (embedder, reranker, app).

## Requirements

- NVIDIA GPU, 24 GB+ VRAM
- Docker 24+ with NVIDIA Container Toolkit
- ~40 GB free disk for model cache

## Setup

```bash
git clone https://github.com/taeyoung-ko/hci-paper-semantic-search.git
cd hci-paper-semantic-search
docker compose up -d embedder reranker          # ~10 min first time
docker compose run --rm app python build_index.py   # ~5–7 min, builds 6 mode-aware indexes
docker compose up -d                             # launch UI at http://localhost:7860
```

The app shows an orange warmup banner for ~30s on first launch.

## Usage

Fill in any of the five component boxes — empty boxes are skipped.

| Box | What to type |
|---|---|
| Background | Research context / domain |
| Gap | Limitation of prior work |
| Solution | System or artifact you propose |
| Approach/Method | Methodology or technique |
| Findings | Experimental results |

Two or more boxes → a **Combined** tab is added (RRF k=60).

**Sliders:** retrieval candidates (default 1000) and final results (default 100).

**Result tabs:** List view (ranked cards) + Graph view (UMAP radial layout, click any node to open DOI). Combined tab is List-only and shows per-mode rank chips on each card.

**Collection:** click ☆ to add a paper. The Accordion below the search shows your collection. Each card has an editable note (autosaves on blur), a source disclosure, and an × to remove.

**Export:** click "Export .bib" in the collection panel. Generates a standard BibTeX file from local metadata, with source/note comments per entry.

## Common commands

```bash
docker compose stop          # pause
docker compose start         # resume
docker compose logs -f       # tail logs
```

## License
Code: MIT. Models retain their original licenses (Apache-2.0).