# HCI Paper Semantic Search

Two-stage semantic search over 13,488 papers from seven major HCI conferences (CHI, UIST, CSCW, DIS, IUI, RecSys, UMAP).

- **Retrieval** — Qwen3-Embedding-0.6B → FAISS top-1000 (default, tunable)
- **Reranking** — Qwen3-Reranker-8B (cross-encoder) → top-100 (default, tunable)

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

```bashnvidia-smi                              # driver + GPU visible
docker --version                        # Docker installed
docker compose version                  # Compose v2 plugin installed
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# GPU passthrough works

If the last command fails, install the NVIDIA Container Toolkit:

```bashsudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

## Setup

### 1. Clone the repository

```bashgit clone https://github.com/taeyoung-ko/hci-paper-semantic-search.git
cd hci-paper-semantic-search

### 2. Bring up the vLLM services

```bashdocker compose up -d embedder reranker

The first run downloads ~16 GB of model weights into a docker-managed volume. Wait until both services report `(healthy)`:

```bashdocker compose ps

This usually takes 5–10 minutes for the embedder and 8–15 minutes for the reranker (8B model + CUDA graph compilation).

### 3. Build the FAISS index

```bashdocker compose run --rm app python build_index.py

Encodes every paper in `data/sigchi_conf_doi.jsonl` (about 1 minute on a recent GPU) and saves the FAISS index into a docker-managed volume. Run this once.

### 4. Launch the search UI

```bashdocker compose up -d

Open <http://localhost:7860> in your browser.

When the app first launches, an orange banner appears at the top of the page:

> ⏳ Warming up the search engine — first launch takes ~30s.

This is a one-time numba/UMAP JIT warmup that runs in the background. The banner disappears automatically (within ~30s) and search becomes available. If you click Search before warmup finishes, the same message is shown in the results area.

## Usage

1. Type a natural-language query in any language (the underlying models are multilingual)
2. Optionally narrow by **Venue** using the checkboxes
3. Adjust the two sliders if you want to:
   - **Retrieval candidates** — how many top-K papers the embedding stage hands to the reranker (default 1000, range 100–5000)
   - **Final results** — how many top-K papers the reranker keeps as the final answer (default 100, range 10–500)
4. Click **Search**

Results appear under two tabs:

- **List** — ranked cards. Each card shows the title (DOI link), authors, venue/year, keywords, abstract, and a colored score badge. The badge color is a linear red→green gradient based on the reranker score.
- **Graph** — UMAP layout. The query sits at the center as a black star. Each paper's distance from the center encodes `1 − rerank_score`, while its angular direction preserves UMAP cluster structure (papers about similar topics cluster together). Node size and color also reflect the reranker score. Hover for metadata; click any node to open its DOI in a new tab.

After warmup, a search typically takes 10–30 seconds (most of it is the reranker scoring 1000 candidates).

## Common commands

```bashStop everything (preserves model cache and index)
docker compose stopResume
docker compose startTear down containers (keeps volumes)
docker compose downWipe everything including model cache (rare; forces re-download)
docker compose down -vWatch logs
docker compose logs -fWatch GPU usage
watch -n 1 nvidia-smi

## ArchitectureBrowser ──► Gradio UI (app)
│
├──► /v1/embeddings   Qwen3-Embedding-0.6B (vLLM)
│       └─► FAISS top-K retrieval (with venue filter)
│
└──► /v1/score        Qwen3-Reranker-8B (vLLM)
└─► top-N results (cross-encoder rescored)
│
├─► List view  (ranked cards)
└─► Graph view (UMAP + radial layout)

## Data

`data/sigchi_conf_doi.jsonl` contains paper metadata (title, abstract, authors, keywords) collected from OpenAlex and Crossref for the seven HCI venues listed above.

## Troubleshooting

**Reranker scores are all near 0.5** — vLLM regression bug. Pin the image to `vllm/vllm-openai:v0.14.0` (the version specified in `compose.yaml`).

**`Restarting (1)` on the app container** — usually a Python dependency conflict. Check logs:
```bashdocker compose logs --tail=80 app

**`chown ... operation not permitted`** — happens on NFS mounts. The `compose.yaml` already uses Docker-managed named volumes (`hf_cache`, `paper_index`) to avoid this.

**Graph tab is blank** — the iframe-embedded plotly view depends on inline scripts. If you see a blank panel, hard-reload the page (Ctrl+Shift+R) and check the browser console for CSP errors.

**First search after long idle hangs for ~30s** — vLLM CUDA graph cache may have been evicted. Subsequent searches will be fast again.

## License

Code: MIT.  
Models retain their original licenses — Qwen3-Embedding and Qwen3-Reranker are released under Apache-2.0.
