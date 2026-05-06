# HCI Paper Semantic Search

Two-stage semantic search over HCI conference papers, with a built-in collector for venue metadata from Crossref and OpenAlex.

- **Similar Papers** — paste an abstract or paper description, find papers like it (`topic` mode).
- **Related Work** — fill in your paper's components (Background / Gap / Solution / Method / Findings); each runs as its own instruction-tuned search and a **Combined** tab fuses the rankings via RRF (k = 60).
- **Manage Data** — pivot table UI to collect papers for any ACM venue × year. Auto-rebuilds search indexes on completion.

Pipeline: embed (Qwen3-Embedding-0.6B) → FAISS top-K → rerank (Qwen3-Reranker-8B cross-encoder) → top-N. Star papers, attach notes, export to `.bib`.

Runs locally via Docker Compose (embedder, reranker, app).

## Requirements

- NVIDIA GPU, 40 GB+ VRAM
- Docker 24+ with NVIDIA Container Toolkit
- ~40 GB free disk for model cache

## Setup

```bash
git clone https://github.com/taeyoung-ko/hci-paper-semantic-search.git
cd hci-paper-semantic-search
docker compose up -d embedder reranker   # ~10 min first run, downloads models
docker compose up -d app                  # API + UI at http://localhost:7860
```

A fresh clone ships **no paper data**. `data/sigchi_conf_doi.jsonl` is built locally. Two ways to bootstrap:

**A. Via UI (recommended).** Open http://localhost:7860 → **Manage Data** tab → enter your email (sent to Crossref / OpenAlex polite pool) → tick the venue × year cells you want → **Collect**. The collector pulls metadata, deduplicates by DOI, appends to the JSONL, and rebuilds all 6 mode-aware indexes automatically.

**B. Manual.** Drop your own `data/sigchi_conf_doi.jsonl` (one paper per line with `doi`, `title`, `abstract`, `authors`, `venue`, `year`, `keywords`), then:

```bash
docker compose run --rm app python build_index.py   # ~5–7 min for 6 mode-aware indexes
```

### GPU memory tuning

Default values work on 40 GB+ GPUs. Adjust via `.env` in the project root:

```env
EMBEDDER_GPU_UTIL=0.05
RERANKER_GPU_UTIL=0.20
```

## Usage

### Similar Papers

Single-box search backed by the `topic` mode (title + abstract embedding). Paste an abstract or short paper description; results are papers most similar to that input.

### Related Work

Fill any of the five component boxes — empty boxes are skipped, filled ones run in parallel as separate instruction-tuned searches.

| Box | What to type |
|---|---|
| Background | Research context / domain |
| Research Gap | Limitation of prior work |
| Solution | System or artifact you propose |
| Approach/Method | Methodology or technique |
| Findings | Experimental results |

Two or more boxes → a **Combined** tab is added (Reciprocal Rank Fusion, k = 60).

### Manage Data

Pivot table of `Venue × Year`. Each cell carries a stem DOI for Crossref proceedings scan; each row optionally carries a PACM ISSN + track for OpenAlex journal scan (e.g., CSCW → ISSN `2573-0142`, track `cscw`). Tick the cells you want, then **Collect**.

Toggle:
- **Skip existing** — if a (venue, year) already has any paper in the JSONL, skip the API scan entirely. Off by default; off rescans every time and merges new DOIs.

Progress is reported per (venue, year). When done, indexes rebuild and the search tabs refresh.

### Filters & options

- **Venue filter** above the query boxes — chips to include/exclude conferences.
- **Advanced** — Retrieval K and Rerank K sliders.

### Results

Ranked cards with rerank score, authors, venue/year, keywords, and full abstract. Combined tab shows per-mode rank chips on each card.

### My Collection

Click ★ on any result to add it. The **My Collection** tab in the top nav lists all starred papers with the same card layout as search results (authors, venue/year, keywords, abstract). Each card includes:

- Inline note textarea (autosave, ~400 ms debounce)
- Selection checkbox (per-card)
- ★ button — click again to remove from the collection

Header toolbar: **Select all** / **Deselect all** / **Export .bib**. The export button label switches between *Export selected (N)* and *Export all (N)* depending on whether anything is selected. Output is a standard BibTeX file with source/note comments per entry.

### Clear

Top-nav **Clear** button wipes search inputs, search results, and the starred collection. Venue filter, slider settings, and Manage Data state are kept.

### Persistence

Query inputs, search results, and collection are saved to `localStorage`. Refreshing the page preserves your session.

## Architecture

```
Browser (React)
  │
  ├── /api/search             ──► FastAPI ──► Qwen3-Embedding-0.6B (vLLM)  ──► FAISS top-K per mode
  │                                       └─► Qwen3-Reranker-8B   (vLLM)  ──► top-N cross-encoder
  ├── /api/add-venues         ──► FastAPI ──► collector ──► Crossref + OpenAlex
  │                                                    └─► JSONL append + reindex
  ├── /api/add-venue/status   ──► poll collection progress
  ├── /api/export-bib         ──► BibTeX file
  └── /api/filter-options     ──► venue list, year range
```

## Common commands

```bash
docker compose stop          # pause
docker compose start         # resume
docker compose down          # tear down (keeps volumes)
docker compose logs -f       # tail logs
```

## License
Code: MIT. Models retain their original licenses (Apache-2.0).
