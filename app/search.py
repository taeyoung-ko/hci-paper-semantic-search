"""Mode-aware retrieval (0.6B) + filter + rerank (8B)."""
import json
import os
from pathlib import Path

import faiss
import numpy as np

from api_clients import embed_query, rerank, MODES, DEFAULT_MODE

INDEX_DIR = Path(os.environ["INDEX_DIR"])

# Lazy-load all mode indexes at import; small enough to fit in memory.
_indexes: dict[str, faiss.Index] = {}
for mode in MODES:
    path = INDEX_DIR / f"sigchi_{mode}.faiss"
    if path.exists():
        _indexes[mode] = faiss.read_index(str(path))
    else:
        print(f"[search] WARNING: missing index for mode={mode}: {path}")

with (INDEX_DIR / "sigchi_meta.jsonl").open(encoding="utf-8") as f:
    _papers = [json.loads(l) for l in f]


def _doc_text(p: dict) -> str:
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def get_filter_options() -> dict:
    venues = sorted({p.get("venue", "") for p in _papers if p.get("venue")})
    years = [p.get("year") for p in _papers if isinstance(p.get("year"), int)]
    if years:
        year_min, year_max = min(years), max(years)
    else:
        year_min, year_max = 2000, 2026
    return {"venues": venues, "year_min": year_min, "year_max": year_max}


def _passes_filter(p: dict, allowed_venues: set[str] | None,
                   year_min: int, year_max: int) -> bool:
    if allowed_venues is not None and p.get("venue") not in allowed_venues:
        return False
    y = p.get("year")
    if not isinstance(y, int):
        return False
    if y < year_min or y > year_max:
        return False
    return True


def search(query: str,
           mode: str = DEFAULT_MODE,
           allowed_venues: list[str] | None = None,
           year_min: int = 0,
           year_max: int = 9999,
           retrieve_k: int = 1000,
           rerank_k: int = 100) -> list[dict]:
    """Two-stage mode-aware semantic search.

    Returns list of paper dicts with rerank_score attached.
    """
    if not query.strip():
        return []
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode not in _indexes:
        raise RuntimeError(
            f"Index for mode={mode} not loaded. Did you run build_index.py?"
        )

    venue_set = set(allowed_venues) if allowed_venues else None
    index = _indexes[mode]

    # Stage 1: query embedding (mode-aware)
    q_vec = np.asarray(embed_query(query, mode=mode), dtype="float32")
    q_norm = q_vec.copy()
    faiss.normalize_L2(q_norm.reshape(1, -1))

    # Stage 2 + 3: FAISS retrieve + filter (with adaptive widening)
    target_survivors = max(rerank_k, 100)
    n_total = index.ntotal
    k = min(retrieve_k, n_total)
    survivors: list[dict] = []

    while True:
        emb_scores, idxs = index.search(q_norm.reshape(1, -1), k)
        survivors = []
        seen: set[int] = set()
        for s, i in zip(emb_scores[0], idxs[0]):
            if i < 0 or int(i) in seen:
                continue
            seen.add(int(i))
            p = _papers[int(i)]
            if not _passes_filter(p, venue_set, year_min, year_max):
                continue
            row = dict(p)
            row["embed_score"] = float(s)
            survivors.append(row)
        if len(survivors) >= target_survivors or k >= n_total:
            break
        k = min(k * 3, n_total)

    if not survivors:
        return []

    # Stage 3: rerank (mode-aware)
    docs = [_doc_text(p) for p in survivors]
    rerank_scores = rerank(query, docs, mode=mode, batch_size=32)
    for p, s in zip(survivors, rerank_scores):
        p["rerank_score"] = s

    # Stage 4: top-K
    survivors.sort(key=lambda x: x["rerank_score"], reverse=True)
    return survivors[:rerank_k]
