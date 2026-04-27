"""retrieval (0.6B) + filter + rerank (8B) 통합 검색."""
import json
import os
from pathlib import Path

import faiss
import numpy as np

from api_clients import embed_query, rerank

INDEX_DIR = Path(os.environ["INDEX_DIR"])

_index = faiss.read_index(str(INDEX_DIR / "sigchi.faiss"))
with (INDEX_DIR / "sigchi_meta.jsonl").open(encoding="utf-8") as f:
    _papers = [json.loads(l) for l in f]

# FAISS의 IndexFlatIP에서 reconstruct로 원본 벡터 복원 가능
# (정규화된 상태로 저장되어 있음)


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
           allowed_venues: list[str] | None = None,
           year_min: int = 0,
           year_max: int = 9999,
           retrieve_k: int = 1000,
           rerank_k: int = 100) -> dict:
    """
    Returns: {
        "query_embedding": np.ndarray (D,),
        "results": [paper dicts with rerank_score, embed_score, _vec],
    }
    The "_vec" field on each paper is its 1024-d normalized embedding,
    used downstream for graph layout. It is NOT meant for serialization.
    """
    if not query.strip():
        return {"query_embedding": None, "results": []}

    venue_set = set(allowed_venues) if allowed_venues else None

    # Stage 1: embed query
    q_vec = np.asarray(embed_query(query), dtype="float32")
    q_norm = q_vec.copy()
    faiss.normalize_L2(q_norm.reshape(1, -1))

    # Stage 2 + 3 + adaptive widening
    target_survivors = max(rerank_k, 100)
    n_total = _index.ntotal
    k = min(retrieve_k, n_total)
    survivors: list[dict] = []

    while True:
        emb_scores, idxs = _index.search(q_norm.reshape(1, -1), k)
        survivors = []
        seen_ids: set[int] = set()
        for s, i in zip(emb_scores[0], idxs[0]):
            if i < 0 or int(i) in seen_ids:
                continue
            seen_ids.add(int(i))
            p = _papers[int(i)]
            if not _passes_filter(p, venue_set, year_min, year_max):
                continue
            row = dict(p)
            row["embed_score"] = float(s)
            row["_idx"] = int(i)
            survivors.append(row)
        if len(survivors) >= target_survivors or k >= n_total:
            break
        k = min(k * 3, n_total)

    if not survivors:
        return {"query_embedding": q_norm, "results": []}

    # Stage 4: rerank
    docs = [_doc_text(p) for p in survivors]
    rerank_scores = rerank(query, docs, batch_size=32)
    for p, s in zip(survivors, rerank_scores):
        p["rerank_score"] = s

    # Stage 5: top-k by rerank_score
    survivors.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = survivors[:rerank_k]

    # Attach embedding vectors for graph layout
    for p in top:
        v = _index.reconstruct(p["_idx"])
        p["_vec"] = np.asarray(v, dtype="float32")

    return {"query_embedding": q_norm, "results": top}
