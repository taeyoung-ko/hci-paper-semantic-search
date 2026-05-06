"""BERTopic-based topic trends over time.

Reuses pre-computed FAISS embeddings (Qwen3-Embedding-0.6B, `topic` mode).
Clusters with HDBSCAN (default in BERTopic), reduces with UMAP,
labels with c-TF-IDF on filtered titles+abstracts.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter

import numpy as np

from search import _indexes, _papers

log = logging.getLogger("trends")

MIN_PAPERS = 30  # below this, BERTopic clustering is too noisy

_cache: dict[tuple, dict] = {}
_cache_lock = threading.Lock()


def _filter_indices(allowed_venues: set[str] | None,
                    year_min: int, year_max: int) -> list[int]:
    out = []
    for i, p in enumerate(_papers):
        y = p.get("year")
        if not isinstance(y, int) or y < year_min or y > year_max:
            continue
        if allowed_venues is not None and p.get("venue") not in allowed_venues:
            continue
        out.append(i)
    return out


def _get_embeddings(indices: list[int]) -> np.ndarray:
    idx = _indexes.get("topic")
    if idx is None:
        raise RuntimeError("topic index not loaded")
    n = idx.ntotal
    all_vecs = idx.reconstruct_n(0, n)
    return np.asarray(all_vecs[indices], dtype="float32")


def _build_text(p: dict) -> str:
    t = (p.get("title") or "").strip()
    a = (p.get("abstract") or "").strip()
    if t and a:
        return f"{t}. {a}"
    return t or a or "(no text)"


def _clean_paper(p: dict) -> dict:
    return {
        "doi": p.get("doi", ""),
        "title": p.get("title", ""),
        "authors": p.get("authors", []),
        "venue": p.get("venue", ""),
        "year": p.get("year"),
        "keywords": p.get("keywords", []),
        "abstract": p.get("abstract", ""),
    }


def build_trends(allowed_venues: list[str] | None,
                 year_min: int, year_max: int,
                 force: bool = False) -> dict:
    venues_t = tuple(sorted(allowed_venues)) if allowed_venues else ()
    cache_key = (venues_t, int(year_min), int(year_max))

    if not force:
        with _cache_lock:
            entry = _cache.get(cache_key)
        if entry is not None:
            log.info("trends cache hit")
            return entry["result"]

    venue_set = set(allowed_venues) if allowed_venues else None
    indices = _filter_indices(venue_set, year_min, year_max)

    if len(indices) < MIN_PAPERS:
        result = {
            "ok": False,
            "error": f"Need at least {MIN_PAPERS} papers; got {len(indices)}.",
            "topics": [],
            "over_time": [],
            "year_range": [year_min, year_max],
        }
        return result

    log.info("trends: clustering %d papers", len(indices))
    embeddings = _get_embeddings(indices)
    docs = [_build_text(_papers[i]) for i in indices]
    timestamps = [_papers[i]["year"] for i in indices]

    # Lazy import — bertopic loads heavy deps (umap, hdbscan)
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )

    model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)

    info = model.get_topic_info()  # DataFrame: Topic, Count, Name, Representation, ...
    topics_list = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # outliers/noise — skip in main view
        words = [w for w, _ in (model.get_topic(tid) or [])][:8]
        topics_list.append({
            "id": tid,
            "name": row.get("Name", str(tid)),
            "count": int(row["Count"]),
            "words": words,
        })

    # topics_over_time: DataFrame [Topic, Words, Frequency, Timestamp, Name]
    tot_df = model.topics_over_time(docs, timestamps,
                                    nr_bins=None,
                                    global_tuning=True,
                                    evolution_tuning=True)
    over_time = []
    for _, row in tot_df.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        ts = row["Timestamp"]
        # Timestamp can be int year or pandas Timestamp depending on input
        try:
            year = int(ts.year) if hasattr(ts, "year") else int(ts)
        except Exception:
            year = int(str(ts)[:4])
        over_time.append({
            "topic_id": tid,
            "year": year,
            "frequency": int(row["Frequency"]),
        })

    # Per-year totals (for normalization on the frontend)
    year_totals = Counter(timestamps)

    # Group papers by their assigned topic for later lookup
    papers_by_topic: dict[int, list] = {}
    for paper_idx, t_id in enumerate(topics):
        if t_id == -1:
            continue
        papers_by_topic.setdefault(int(t_id), []).append(
            _clean_paper(_papers[indices[paper_idx]])
        )
    # Sort papers within each topic by year (desc), then title
    for tid in papers_by_topic:
        papers_by_topic[tid].sort(
            key=lambda p: (-(p.get("year") or 0), p.get("title", ""))
        )

    result = {
        "ok": True,
        "topics": topics_list,
        "over_time": over_time,
        "year_totals": [
            {"year": int(y), "total": int(c)}
            for y, c in sorted(year_totals.items())
        ],
        "n_papers": len(indices),
        "n_outliers": sum(1 for t in topics if t == -1),
        "year_range": [year_min, year_max],
    }

    with _cache_lock:
        _cache[cache_key] = {"result": result, "papers_by_topic": papers_by_topic}
    log.info("trends: %d topics, %d outliers (%d papers)",
             len(topics_list), result["n_outliers"], len(indices))
    return result


def get_topic_papers(allowed_venues: list[str] | None,
                     year_min: int, year_max: int,
                     topic_id: int) -> dict:
    venues_t = tuple(sorted(allowed_venues)) if allowed_venues else ()
    cache_key = (venues_t, int(year_min), int(year_max))
    with _cache_lock:
        entry = _cache.get(cache_key)
    if entry is None:
        return {"ok": False, "error": "No cached trends. Run trends first.", "papers": []}
    papers = entry.get("papers_by_topic", {}).get(int(topic_id), [])
    return {"ok": True, "papers": papers}


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
