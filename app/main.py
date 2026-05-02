"""FastAPI backend for HCI Paper Semantic Search."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from search import search, get_filter_options
from bib_export import write_bib_to_tempfile

app = FastAPI(title="HCI Paper Semantic Search")

# ── Filter options (loaded once) ──

_opts = get_filter_options()


# ── Per-mode LRU search cache ──

CACHE_PER_MODE = 5
_cache: dict[str, dict] = {}


def _cache_key(mode, qtext, venues, retrieve_k, rerank_k):
    venues_t = tuple(sorted(venues)) if venues else ()
    return (mode, qtext, venues_t, int(retrieve_k), int(rerank_k))


def _cache_get(key):
    mode = key[0]
    bucket = _cache.get(mode)
    if not bucket:
        return None
    if key in bucket:
        val = bucket.pop(key)
        bucket[key] = val
        return val
    return None


def _cache_put(key, value):
    mode = key[0]
    bucket = _cache.setdefault(mode, {})
    if key in bucket:
        bucket.pop(key)
    bucket[key] = value
    while len(bucket) > CACHE_PER_MODE:
        oldest = next(iter(bucket))
        bucket.pop(oldest)


def _search_cached(qtext, mode, allowed_venues, retrieve_k, rerank_k):
    key = _cache_key(mode, qtext, allowed_venues, retrieve_k, rerank_k)
    hit = _cache_get(key)
    if hit is not None:
        print(f"[cache hit] mode={mode}")
        return hit
    print(f"[cache miss] mode={mode}")
    res = search(qtext, mode=mode, allowed_venues=allowed_venues,
                 retrieve_k=retrieve_k, rerank_k=rerank_k)
    _cache_put(key, res)
    return res


@app.get("/api/filter-options")
def filter_options():
    return _opts


# ── Search ──

RRF_K = 60


class SearchRequest(BaseModel):
    background: str = ""
    gap: str = ""
    solution: str = ""
    method: str = ""
    findings: str = ""
    venues: list[str] = Field(default_factory=list)
    retrieve_k: int = 1000
    rerank_k: int = 100


class PaperResult(BaseModel):
    doi: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    venue: str = ""
    year: Optional[int] = None
    keywords: list[str] = Field(default_factory=list)
    abstract: str = ""
    rerank_score: float = 0.0
    embed_score: float = 0.0


class CombinedPaperResult(PaperResult):
    rrf_score: float = 0.0
    rrf_modes: dict[str, int] = Field(default_factory=dict)


class SearchTab(BaseModel):
    label: str
    mode: str
    query: str
    results: list[PaperResult]


class SearchResponse(BaseModel):
    tabs: list[SearchTab]
    combined: Optional[dict] = None
    errors: list[str] = Field(default_factory=list)


def _clean_result(p: dict) -> dict:
    """Strip internal fields, keep only serializable paper data."""
    return {
        "doi": p.get("doi", ""),
        "title": p.get("title", ""),
        "authors": p.get("authors", []),
        "venue": p.get("venue", ""),
        "year": p.get("year"),
        "keywords": p.get("keywords", []),
        "abstract": p.get("abstract", ""),
        "rerank_score": float(p.get("rerank_score", 0.0)),
        "embed_score": float(p.get("embed_score", 0.0)),
    }


def _combine_rrf(per_mode_results: list[tuple[str, list[dict]]],
                 rerank_k: int) -> list[dict]:
    """Reciprocal Rank Fusion over multiple mode rankings."""
    by_doi: dict[str, dict] = {}
    for mode_label, results in per_mode_results:
        for rank_idx, p in enumerate(results, start=1):
            doi = p.get("doi") or p.get("title")
            if not doi:
                continue
            entry = by_doi.get(doi)
            if entry is None:
                entry = _clean_result(p)
                entry["rrf_score"] = 0.0
                entry["rrf_modes"] = {}
                by_doi[doi] = entry
            entry["rrf_score"] += 1.0 / (RRF_K + rank_idx)
            entry["rrf_modes"][mode_label] = rank_idx

    fused = sorted(by_doi.values(), key=lambda d: d["rrf_score"], reverse=True)
    return fused[:max(1, rerank_k)]


@app.post("/api/search")
def do_search(req: SearchRequest):
    components = [
        ("background", "Background", req.background.strip()),
        ("gap", "Gap", req.gap.strip()),
        ("solution", "Solution", req.solution.strip()),
        ("method", "Approach/Method", req.method.strip()),
        ("findings", "Findings", req.findings.strip()),
    ]
    active = [(m, lbl, q) for m, lbl, q in components if q]

    if not active:
        return SearchResponse(tabs=[], errors=["Fill in at least one component box."])

    venues = req.venues if req.venues else None
    rk = max(1, req.rerank_k)
    rt = max(rk, req.retrieve_k)

    tabs: list[dict] = []
    errors: list[str] = []

    def _one_search(item):
        mode, label, qtext = item
        try:
            results = _search_cached(qtext, mode, venues,
                                     retrieve_k=rt, rerank_k=rk)
            return ("ok", label, mode, qtext, results, None)
        except Exception as e:
            return ("err", label, mode, qtext, None, str(e))

    with ThreadPoolExecutor(max_workers=len(active)) as ex:
        futures = {ex.submit(_one_search, item): idx
                   for idx, item in enumerate(active)}
        results_by_idx = {}
        for fut in futures:
            idx = futures[fut]
            results_by_idx[idx] = fut.result()

    successful = []
    for i in range(len(active)):
        status, label, mode, qtext, results, err = results_by_idx[i]
        if status == "ok":
            cleaned = [_clean_result(p) for p in results]
            tabs.append({
                "label": f"{label} ({len(cleaned)})",
                "mode": mode,
                "query": qtext,
                "results": cleaned,
            })
            successful.append((label, results))
        else:
            errors.append(f"{label}: {err}")

    combined = None
    if len(successful) >= 2:
        fused = _combine_rrf(successful, rk)
        combined = {
            "label": f"Combined ({len(fused)})",
            "results": fused,
        }

    return {"tabs": tabs, "combined": combined, "errors": errors}


# ── BibTeX export ──

class ExportRequest(BaseModel):
    collection: list[dict] = Field(default_factory=list)


@app.post("/api/export-bib")
def export_bib(req: ExportRequest):
    path = write_bib_to_tempfile(req.collection)
    return FileResponse(
        path,
        media_type="application/x-bibtex",
        filename=Path(path).name,
    )


# ── Serve React build ──

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
