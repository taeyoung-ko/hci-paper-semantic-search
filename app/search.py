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


def _doc_text(p: dict) -> str:
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def get_filter_options() -> dict:
    """UI 초기화에 사용할 필터 옵션 데이터."""
    venues = sorted({p.get("venue", "") for p in _papers if p.get("venue")})
    years = [p.get("year") for p in _papers if isinstance(p.get("year"), int)]
    if years:
        year_min, year_max = min(years), max(years)
    else:
        year_min, year_max = 2000, 2026
    return {
        "venues": venues,
        "year_min": year_min,
        "year_max": year_max,
    }


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
           rerank_k: int = 100) -> list[dict]:
    """
    Pre-filter semantics:
      1) embed query
      2) FAISS retrieve top-K
      3) keep only papers passing (venue ∩ year) filter
      4) rerank survivors with 8B
      5) return top rerank_k

    If filter is very narrow and survivors < rerank_k after step 3,
    we widen the FAISS retrieval up to the entire index to give the
    reranker more material to score.
    """
    if not query.strip():
        return []

    venue_set = set(allowed_venues) if allowed_venues else None

    # Stage 1: embed
    q_emb = np.asarray([embed_query(query)], dtype="float32")
    faiss.normalize_L2(q_emb)

    # Stage 2 + 3 + (adaptive widening): keep retrieving until enough survivors
    # or the index is exhausted.
    target_survivors = max(rerank_k, 100)
    n_total = _index.ntotal
    k = min(retrieve_k, n_total)
    survivors: list[dict] = []
    seen_ids: set[int] = set()

    while True:
        emb_scores, idxs = _index.search(q_emb, k)
        survivors = []
        seen_ids = set()
        for s, i in zip(emb_scores[0], idxs[0]):
            if i < 0 or int(i) in seen_ids:
                continue
            seen_ids.add(int(i))
            p = _papers[int(i)]
            if not _passes_filter(p, venue_set, year_min, year_max):
                continue
            row = dict(p)
            row["embed_score"] = float(s)
            survivors.append(row)
        if len(survivors) >= target_survivors or k >= n_total:
            break
        # Widen the net.
        k = min(k * 3, n_total)

    if not survivors:
        return []

    # Stage 4: rerank
    docs = [_doc_text(p) for p in survivors]
    rerank_scores = rerank(query, docs, batch_size=32)
    for p, s in zip(survivors, rerank_scores):
        p["rerank_score"] = s

    # Stage 5: top-K
    survivors.sort(key=lambda x: x["rerank_score"], reverse=True)
    return survivors[:rerank_k]
