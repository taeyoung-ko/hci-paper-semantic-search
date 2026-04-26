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
