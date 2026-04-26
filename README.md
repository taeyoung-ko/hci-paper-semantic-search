# HCI Paper Semantic Search

Two-stage semantic search over 13,488 papers from seven major HCI conferences (CHI, UIST, CSCW, DIS, IUI, RecSys, UMAP).

- **Retrieval** — Qwen3-Embedding-0.6B → FAISS top-1000
- **Reranking** — Qwen3-Reranker-8B (cross-encoder) → top-100

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
git clone https://github.com/USERNAME/REPO.git
cd REPO
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

### 3. Build the FAISS index

```bash
docker compose run --rm app python build_index.py
```

Encodes every paper in `data/sigchi_conf_doi.jsonl` (about 1 minute on a recent GPU) and saves the FAISS index into a docker-managed volume. Run this once.

### 4. Launch the search UI

```bash
docker compose up -d
```

Open <http://localhost:7860> in your browser.

## Usage

1. Type a natural-language query in any language (the underlying models are multilingual)
2. Optionally narrow by venue using the checkboxes
3. Click **Search**

The first search after a fresh start takes 30–60 seconds (vLLM warming up); subsequent searches finish in 10–30 seconds. Results are color-coded by reranker confidence (red → green, 0 → 1).

## Common commands

```bash
# Stop everything (preserves model cache and index)
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
              │       └─► FAISS top-1000 retrieval (with venue filter)
              │
              └──► /v1/score        Qwen3-Reranker-8B (vLLM)
                       └─► top-100 results (cross-encoder rescored)
```

## Data

`data/sigchi_conf_doi.jsonl` contains paper metadata (title, abstract, authors, keywords) collected from OpenAlex and Crossref for the seven HCI venues listed above.

## Troubleshooting

**Reranker scores are all near 0.5** — vLLM regression bug. Pin the image to `vllm/vllm-openai:v0.14.0` (the version specified in `compose.yaml`).

**`Restarting (1)` on the app container** — usually a Python dependency conflict. Check logs:
```bash
docker compose logs --tail=80 app
```

**`chown ... operation not permitted`** — happens on NFS mounts. The `compose.yaml` already uses Docker-managed named volumes (`hf_cache`, `paper_index`) to avoid this.

**First search hangs for over a minute** — normal during initial CUDA graph compilation. Subsequent searches will be fast.

## License

Code: MIT.  
Models retain their original licenses — Qwen3-Embedding and Qwen3-Reranker are released under Apache-2.0.
