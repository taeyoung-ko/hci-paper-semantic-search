# HCI Paper Semantic Search

Two-stage semantic search over HCI conference papers.
- Embed (Qwen3-Embedding-0.6B) → FAISS top-K
- Rerank (Qwen3-Reranker-8B cross-encoder) → top-N

Split your query into research components (Background / Research Gap / Solution / Approach·Method / Findings). Each filled box runs as its own instruction-tuned search. With two or more boxes filled, a **Combined** tab fuses the rankings via RRF. Star papers to build a collection, attach notes, and export to `.bib`.

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

### GPU memory tuning

Default values work on 24 GB+ GPUs. Adjust via `.env` in the project root:

```env
EMBEDDER_GPU_UTIL=0.05
RERANKER_GPU_UTIL=0.20
```

## Usage

Fill in any of the five component boxes — empty boxes are skipped.

| Box | What to type |
|---|---|
| Background | Research context / domain |
| Research Gap | Limitation of prior work |
| Solution | System or artifact you propose |
| Approach/Method | Methodology or technique |
| Findings | Experimental results |

Two or more boxes → a **Combined** tab is added (RRF k=60).

**Options:** Venue filter above the query boxes. Retrieval K and Rerank K sliders under Advanced.

**Results:** Ranked cards with rerank score, authors, venue/year, keywords, and full abstract. Combined tab shows per-mode rank chips on each card.

**Collection:** Click ★ on any result to add it. Open the collection drawer from the right-side tab. Each card has an editable note (autosaved) and a remove button.

**Export:** Click "Export .bib" in the collection drawer. Generates a standard BibTeX file with source/note comments per entry.

**Persistence:** Query inputs, search results, and collection are saved to localStorage. Refreshing the page preserves your session.

## Architecture

```
Browser (React)
  │
  ├── POST /api/search ──► FastAPI backend
  │                           ├──► /v1/embeddings   Qwen3-Embedding-0.6B (vLLM)
  │                           │       └─► FAISS top-K per mode
  │                           └──► /v1/score        Qwen3-Reranker-8B (vLLM)
  │                                    └─► top-N cross-encoder rescored
  ├── POST /api/export-bib
  └── GET  /api/filter-options
```

## Data

`data/sigchi_conf_doi.jsonl` contains paper metadata (title, abstract, authors, keywords) collected from OpenAlex and Crossref. Use `get_metadata.py` to regenerate or extend the dataset, and `retry.py` to re-fetch missing abstracts.

## Common commands

```bash
docker compose stop          # pause
docker compose start         # resume
docker compose down          # tear down (keeps volumes)
docker compose logs -f       # tail logs
```

## License
Code: MIT. Models retain their original licenses (Apache-2.0).
