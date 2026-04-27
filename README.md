```markdown
# HCI Paper Semantic Search

Two-stage semantic search over 13,488 papers from seven major HCI conferences (CHI, UIST, CSCW, DIS, IUI, RecSys, UMAP).

- **Retrieval** -- Qwen3-Embedding-0.6B → FAISS top-K (default 1000, tunable)
- **Reranking** -- Qwen3-Reranker-8B (cross-encoder) → top-N (default 100, tunable)

The query side is split into five research components (Background, Gap, Solution, Approach/Method, Findings). Each component you fill in is matched against a separate, instruction-tuned index for that aspect of the paper, so a method-only query ranks papers by methodological similarity even when the domain is different.

Local deployment via Docker Compose with three services: `embedder` and `reranker` (both vLLM), plus `app` (Gradio UI + FAISS).

## Requirements

### Hardware

- NVIDIA GPU with **24 GB+ VRAM** (32 GB+ recommended)
  - Tested on RTX PRO 6000 Blackwell (96 GB)
  - Should also work on RTX 3090 / 4090 (24 GB), A6000 (48 GB), A100 (40/80 GB)
- ~40 GB free disk for HuggingFace model cache
- Linux host (Ubuntu 22.04 or 24.04 recommended)

### Software

- NVIDIA driver **R555+** (CUDA 12.8 compatible)
- Docker **24+**
- Docker Compose **v2.40+**
- NVIDIA Container Toolkit configured for Docker

Verify your setup:

```bash
nvidia-smi                              # driver + GPU visible
docker --version                        # Docker installed
docker compose version                  # Compose v2 plugin installed
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
                                        # GPU passthrough works
```

If the last command fails, install the NVIDIA Container Toolkit:

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/taeyoung-ko/hci-paper-semantic-search.git
cd hci-paper-semantic-search
```

### 2. Bring up the vLLM services

```bash
docker compose up -d embedder reranker
```

The first run downloads ~16 GB of model weights into a docker-managed volume. Wait until both services report `(healthy)`:

```bash
docker compose ps
```

This usually takes 5–10 minutes for the embedder and 8–15 minutes for the reranker (8B model + CUDA graph compilation).

### 3. Build the FAISS indexes

```bash
docker compose run --rm app python build_index.py
```

Builds **six** instruction-aware FAISS indexes (one per search mode: Topic / Background / Gap / Solution / Method / Findings) over `data/sigchi_conf_doi.jsonl`. Takes about 5–7 minutes on a recent GPU. Run this once, or whenever the data file changes.

If you previously built only a single-mode index from an older version of this repo, re-run this step — the search code now expects all six index files.

### 4. Launch the search UI

```bash
docker compose up -d
```

Open <http://localhost:7860> in your browser.

When the app first launches, an orange banner appears at the top of the page:

> Warming up the search engine — first launch takes ~30s.

This is a one-time numba/UMAP JIT warmup that runs in the background. The banner disappears automatically (within ~30s) and search becomes available. If you click Search before warmup finishes, the same message is shown in the results area.

## Usage

The query is split into five component boxes that map to different aspects of a paper. Fill only the boxes you care about — empty boxes are skipped.

| Box | What to type |
|---|---|
| **Background** | The research context or domain your work sits in |
| **Gap** | The limitation of prior work that your research addresses |
| **Solution** | The system, tool, or artifact you propose |
| **Approach/Method** | The methodology or technique you use |
| **Findings** | The experimental results or main findings of your study |

Behavior:

- **One box filled** — the system searches only that mode and shows one result tab.
- **Two or more boxes filled** — each filled box runs as its own mode-specific search, plus an additional **Overall** search that combines all of them. You get one result tab per search.
- **No boxes filled** — a hint is shown and nothing is searched.

Other controls:

- **Venue** — limit results to specific conferences. All seven are selected by default.
- **Retrieval candidates** (slider, default 1000, range 100–5000) — how many top-K papers the embedding stage hands to the reranker.
- **Final results** (slider, default 100, range 10–500) — how many top-K papers the reranker keeps as the final answer.

Each result tab contains two subtabs:

- **List** — ranked cards. Each card shows the title (DOI link), authors, venue/year, keywords, abstract, and a colored score badge. Badge color is a linear red→green gradient based on the reranker score.
- **Graph** — UMAP layout. The query sits at the center as a black star. Each paper's distance from the center encodes `1 − rerank_score`, while its angular direction preserves UMAP cluster structure (papers about similar topics cluster together). Node size and color also reflect the reranker score. Hover for metadata; click any node to open its DOI in a new tab.

After warmup, a single-component search typically takes 10–30 seconds. Multi-component searches run all modes in parallel via vLLM continuous batching, so filling five boxes takes roughly 1.5–2× a single search rather than 5×.

## Common commands

```bash
# Stop everything (preserves model cache and indexes)
docker compose stop

# Resume
docker compose start

# Tear down containers (keeps volumes)
docker compose down

# Wipe everything including model cache (rare; forces re-download)
docker compose down -v

# Watch logs
docker compose logs -f

# Watch GPU usage
watch -n 1 nvidia-smi
```

## Architecture

```
Browser ──► Gradio UI (app)
              │
              ├──► /v1/embeddings   Qwen3-Embedding-0.6B (vLLM)
              │       └─► FAISS top-K retrieval per mode (with venue filter)
              │
              └──► /v1/score        Qwen3-Reranker-8B (vLLM)
                       └─► top-N results (cross-encoder rescored)
                              │
                              ├─► List view  (ranked cards)
                              └─► Graph view (UMAP + radial layout)
```

Each search mode (Topic / Background / Gap / Solution / Method / Findings) has:

- Its own document-side embedding instruction used at index-build time
- Its own query-side embedding instruction used at search time
- Its own reranker instruction used during cross-encoder rescoring

The same paper text (title + abstract) is encoded six times with different instructions, producing six FAISS indexes. At search time, the mode determined by which input box you used picks both the index and the reranker prompt.

## Data

`data/sigchi_conf_doi.jsonl` contains paper metadata (title, abstract, authors, keywords) collected from OpenAlex and Crossref for the seven HCI venues listed above.

## Troubleshooting

**Reranker scores are all near 0.5** — vLLM regression bug. Pin the image to `vllm/vllm-openai:v0.14.0` (the version specified in `compose.yaml`).

**`Restarting (1)` on the app container** — usually a Python dependency conflict. Check logs:

```bash
docker compose logs --tail=80 app
```

**`chown ... operation not permitted`** — happens on NFS mounts. The `compose.yaml` already uses Docker-managed named volumes (`hf_cache`, `paper_index`) to avoid this.

**Search returns "Index for mode=X not loaded"** — the mode-aware indexes haven't been built yet. Re-run `docker compose run --rm app python build_index.py`.

**Graph tab is blank** — the iframe-embedded plotly view depends on inline scripts. If you see a blank panel, hard-reload the page (Ctrl+Shift+R) and check the browser console for errors.

**First search after long idle hangs for ~30s** — vLLM CUDA graph cache may have been evicted. Subsequent searches will be fast again.

## License

Code: MIT.
Models retain their original licenses — Qwen3-Embedding and Qwen3-Reranker are released under Apache-2.0.
```
