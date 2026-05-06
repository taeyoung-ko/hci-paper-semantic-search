"""FastAPI backend for HCI Paper Semantic Search."""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from search import search, get_filter_options, reload_indexes
from bib_export import write_bib_to_tempfile

app = FastAPI(title="HCI Paper Semantic Search")

DATA_PATH = Path(os.environ["DATA_PATH"])
INDEX_DIR = Path(os.environ["INDEX_DIR"])


@app.get("/api/filter-options")
def filter_options():
    return get_filter_options()


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


# ── Search ──

RRF_K = 60


class SearchRequest(BaseModel):
    topic: str = ""
    background: str = ""
    gap: str = ""
    solution: str = ""
    method: str = ""
    findings: str = ""
    venues: list[str] = Field(default_factory=list)
    retrieve_k: int = 1000
    rerank_k: int = 100


def _clean_result(p: dict) -> dict:
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


def _combine_rrf(per_mode_results, rerank_k):
    by_doi = {}
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
        ("topic", "Topic", req.topic.strip()),
        ("background", "Background", req.background.strip()),
        ("gap", "Gap", req.gap.strip()),
        ("solution", "Solution", req.solution.strip()),
        ("method", "Approach/Method", req.method.strip()),
        ("findings", "Findings", req.findings.strip()),
    ]
    active = [(m, lbl, q) for m, lbl, q in components if q]

    if not active:
        return {"tabs": [], "combined": None,
                "errors": ["Fill in at least one component box."]}

    venues = req.venues if req.venues else None
    rk = max(1, req.rerank_k)
    rt = max(rk, req.retrieve_k)

    tabs = []
    errors = []

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
                "mode": mode, "query": qtext, "results": cleaned,
            })
            successful.append((label, results))
        else:
            errors.append(f"{label}: {err}")

    combined = None
    if len(successful) >= 2:
        fused = _combine_rrf(successful, rk)
        combined = {"label": f"Combined ({len(fused)})", "results": fused}

    return {"tabs": tabs, "combined": combined, "errors": errors}


# ── BibTeX export ──

class ExportRequest(BaseModel):
    collection: list[dict] = Field(default_factory=list)


@app.post("/api/export-bib")
def export_bib(req: ExportRequest):
    path = write_bib_to_tempfile(req.collection)
    return FileResponse(path, media_type="application/x-bibtex",
                        filename=Path(path).name)


# ── Add venues (batch) ──

class VenueEntry(BaseModel):
    venue: str
    year: int
    stem_doi: str = ""
    pacm_issn: str = ""
    pacm_track: str = ""


class AddVenuesRequest(BaseModel):
    email: str
    entries: list[VenueEntry]
    skip_existing: bool = False


_task_status = {
    "running": False,
    "done": False,
    "error": None,
    "overall_current": 0,
    "overall_total": 0,
    "overall_label": "",
    "step_label": "",
    "step_progress": None,
    "step_detail": "",
}
_task_lock = threading.Lock()


def _run_add_venues(email: str, entries: list[dict], skip_existing: bool = False):
    global _task_status
    from collector import collect_papers
    from build_index import main as rebuild_index

    def _set(**kw):
        with _task_lock:
            _task_status.update(kw)

    try:
        existing_dois = set()
        existing_venue_years = set()
        if DATA_PATH.exists():
            with DATA_PATH.open(encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line)
                    d = p.get("doi", "").lower()
                    if d:
                        existing_dois.add(d)
                    v = (p.get("venue") or "").strip().lower()
                    y = p.get("year")
                    if v and isinstance(y, int):
                        existing_venue_years.add((v, y))

        total_new = 0
        total = len(entries)
        prefix_cache: dict = {}

        for i, entry in enumerate(entries):
            venue = entry["venue"]
            year = entry["year"]
            stem_doi = entry.get("stem_doi", "")
            pacm_issn = entry.get("pacm_issn", "")
            pacm_track = entry.get("pacm_track", "")
            label = f"{venue} {year}"

            _set(overall_current=i, overall_total=total, overall_label=label,
                 step_label="Starting", step_progress=None, step_detail="")

            if skip_existing and (venue.strip().lower(), year) in existing_venue_years:
                _set(step_label="Skipped (already collected)",
                     step_progress=1.0, step_detail=label)
                continue

            def on_step(step_label, progress, detail):
                _set(step_label=step_label, step_progress=progress,
                     step_detail=detail)

            try:
                papers = collect_papers(venue, year, stem_doi,
                                        pacm_issn, pacm_track,
                                        email, on_step=on_step,
                                        prefix_cache=prefix_cache)
            except Exception as e:
                _set(step_label=f"Error: {e}", step_progress=None, step_detail="")
                continue

            new_papers = [p for p in papers
                          if p.get("doi", "").lower() not in existing_dois]

            for p in new_papers:
                d = p.get("doi", "").lower()
                if d:
                    existing_dois.add(d)

            if new_papers:
                with DATA_PATH.open("a", encoding="utf-8") as f:
                    for p in new_papers:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                total_new += len(new_papers)

            _set(step_label="Done", step_progress=1.0,
                 step_detail=f"{len(new_papers)} new / {len(papers)} total")

        if total_new == 0:
            _set(overall_current=total, overall_label="No new papers",
                 step_label="Skipping index rebuild", step_progress=1.0,
                 step_detail="")
        else:
            _set(overall_current=total, overall_label="Rebuilding indexes",
                 step_label=f"{total_new} new papers — rebuilding...",
                 step_progress=None, step_detail="")
            rebuild_index()

            reload_indexes()
            _cache.clear()
            _set(step_label="Complete", step_progress=1.0,
                 step_detail=f"{total_new} papers indexed")

    except Exception as e:
        _set(error=str(e), step_label=f"Fatal: {e}")
    finally:
        _set(running=False, done=True)


@app.post("/api/add-venues")
def add_venues(req: AddVenuesRequest):
    global _task_status
    with _task_lock:
        if _task_status["running"]:
            return {"ok": False, "error": "Already running. Wait for completion."}
        _task_status = {
            "running": True,
            "done": False,
            "error": None,
            "overall_current": 0,
            "overall_total": len(req.entries),
            "overall_label": "",
            "step_label": "Starting...",
            "step_progress": None,
            "step_detail": "",
        }

    entries = [e.dict() for e in req.entries]
    t = threading.Thread(target=_run_add_venues,
                         args=(req.email, entries, req.skip_existing),
                         daemon=True)
    t.start()
    return {"ok": True}


@app.get("/api/add-venue/status")
def add_venue_status():
    with _task_lock:
        return dict(_task_status)


# ── Serve React build ──

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
