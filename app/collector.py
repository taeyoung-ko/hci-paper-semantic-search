"""Paper metadata collector — Crossref prefix scan + OpenAlex PACM collection."""
from __future__ import annotations
import json
import logging
import time

import requests

OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org"
PER_PAGE_OA = 200
PER_PAGE_CR = 1000
SLEEP = 0.1
OA_BATCH = 50

log = logging.getLogger("collector")

# PACM HCI issue → track mapping for years with numeric-only issue fields.
PACM_HCI_TRACK_BY_YEAR_ISSUE: dict[int, dict[str, str]] = {
    2025: {
        "2": "CSCW",
        "7": "CSCW",
    },
    # Add new years as data becomes available
}


# ── HTTP clients ──

class OpenAlex:
    def __init__(self, email: str):
        self.email = email
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"sigchi-collector/1.0 (mailto:{email})"

    def _get(self, path, params=None, retries=4):
        params = dict(params or {})
        params["mailto"] = self.email
        url = f"{OPENALEX_BASE}{path}"
        for attempt in range(retries):
            try:
                r = self.s.get(url, params=params, timeout=60)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return {}
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                return {}
            except requests.RequestException:
                time.sleep(2 ** attempt)
        return {}

    def get_source_by_issn(self, issn):
        d = self._get("/sources", {"filter": f"issn:{issn}", "per-page": 1})
        results = d.get("results") or []
        return results[0] if results else {}

    def list_works(self, filter_str):
        cursor = "*"
        while cursor:
            d = self._get("/works", {
                "filter": filter_str,
                "per-page": PER_PAGE_OA,
                "cursor": cursor,
                "select": ("id,doi,title,authorships,abstract_inverted_index,"
                           "biblio,publication_year,type,is_paratext,keywords"),
            })
            for w in d.get("results", []):
                yield w
            cursor = d.get("meta", {}).get("next_cursor")
            time.sleep(SLEEP)

    def batch_lookup_dois(self, dois, on_batch=None):
        out = {}
        unique = sorted(set(d.lower() for d in dois if d))
        total_batches = (len(unique) + OA_BATCH - 1) // OA_BATCH
        for batch_idx in range(0, len(unique), OA_BATCH):
            chunk = unique[batch_idx:batch_idx + OA_BATCH]
            f = "doi:" + "|".join(chunk)
            cursor = "*"
            while cursor:
                d = self._get("/works", {
                    "filter": f,
                    "per-page": PER_PAGE_OA,
                    "cursor": cursor,
                    "select": ("id,doi,title,authorships,abstract_inverted_index,"
                               "biblio,publication_year,type,is_paratext,keywords"),
                })
                for w in d.get("results", []):
                    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                    if doi:
                        out[doi] = w
                cursor = d.get("meta", {}).get("next_cursor")
                time.sleep(SLEEP)
            if on_batch:
                done = min(batch_idx // OA_BATCH + 1, total_batches)
                on_batch(done, total_batches)
        return out


class Crossref:
    def __init__(self, email: str):
        self.email = email
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"sigchi-collector/1.0 (mailto:{email})"

    def _get(self, url, params=None, retries=4):
        params = dict(params or {})
        params.setdefault("mailto", self.email)
        for attempt in range(retries):
            try:
                r = self.s.get(url, params=params, timeout=120)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return {}
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                return {}
            except requests.RequestException:
                time.sleep(2 ** attempt)
        return {}

    def scan_prefix_year(self, prefix, year, on_page=None):
        cursor = "*"
        items = []
        while cursor:
            d = self._get(f"{CROSSREF_BASE}/works", {
                "filter": (f"prefix:{prefix},type:proceedings-article,"
                           f"from-pub-date:{year},until-pub-date:{year}"),
                "rows": PER_PAGE_CR,
                "cursor": cursor,
                "select": "DOI,title,author,container-title,page",
            })
            msg = d.get("message", {})
            page_items = msg.get("items", [])
            if not page_items:
                break
            for it in page_items:
                items.append({
                    "DOI": (it.get("DOI") or "").lower(),
                    "title": it.get("title"),
                    "author": it.get("author"),
                })
            if on_page:
                on_page(len(items))
            cursor = msg.get("next-cursor")
            if not cursor:
                break
            time.sleep(SLEEP)
        return items


# ── Metadata extraction ──

def reconstruct_abstract(inverted):
    if not inverted:
        return ""
    pos = []
    for w, idxs in inverted.items():
        for i in idxs:
            pos.append((i, w))
    pos.sort()
    return " ".join(w for _, w in pos)


def doi_from_oa(w):
    d = w.get("doi") or ""
    return d[len("https://doi.org/"):] if d.startswith("https://doi.org/") else d


def authors_from_oa(w):
    return [a.get("author", {}).get("display_name", "")
            for a in (w.get("authorships") or [])
            if a.get("author", {}).get("display_name")]


def keywords_from_oa(w):
    return [k.get("display_name", "") for k in (w.get("keywords") or [])
            if k.get("display_name")]


def is_paratext(w):
    return bool(w.get("is_paratext")) or w.get("type") == "paratext"


def authors_from_cr(it):
    out = []
    for a in (it.get("author") or []):
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip() if (given or family) else a.get("name", "")
        if name:
            out.append(name)
    return out


def title_from_cr(it):
    t = it.get("title") or [""]
    return (t[0] if t else "").replace("\n", " ").strip()


def oa_to_row(w, base):
    return {
        **base,
        "doi": doi_from_oa(w),
        "title": (w.get("title") or "").replace("\n", " ").strip(),
        "authors": authors_from_oa(w),
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index"))
                       .replace("\n", " ").strip(),
        "keywords": keywords_from_oa(w),
    }


# ── Track filtering ──

def is_pacm_hci_track(year: int, issue: str, track: str) -> bool:
    track_clean = track.upper().replace(" ", "").strip()
    issue_clean = (issue or "").upper().replace(" ", "").strip()
    if not issue_clean:
        return False
    # 2018-2024: issue starts with track name (e.g., "CSCW1", "CSCW2")
    if issue_clean.startswith(track_clean):
        return True
    # 2025+: explicit (year, issue) → track lookup
    return PACM_HCI_TRACK_BY_YEAR_ISSUE.get(year, {}).get(issue) == track_clean


# ── Collection: Crossref prefix scan ──

def _collect_proceedings(venue, year, stem_doi, email, oa, on_step, prefix_cache=None):
    cr = Crossref(email)
    base = {"venue": venue, "year": year}
    prefix = stem_doi.split("/", 1)[0]
    needle = stem_doi.lower() + "."
    cache_key = (prefix, year)

    if prefix_cache is not None and cache_key in prefix_cache:
        all_items = prefix_cache[cache_key]
        on_step("Crossref (cached)", None, f"reusing {len(all_items)} items")
    else:
        on_step("Scanning Crossref", None, "0 items")

        def _on_page(count):
            on_step("Scanning Crossref", None, f"{count} items")

        all_items = cr.scan_prefix_year(prefix, year, on_page=_on_page)
        if prefix_cache is not None:
            prefix_cache[cache_key] = all_items

    matched = [it for it in all_items if it["DOI"].startswith(needle)]
    on_step("Scanning Crossref", 1.0, f"{len(matched)} papers matched")

    if not matched:
        return []

    dois = [it["DOI"] for it in matched if it.get("DOI")]
    on_step("Enriching metadata", 0.0, f"0 / {len(dois)} DOIs")

    def _on_batch(done, total):
        on_step("Enriching metadata", done / total,
                f"{min(done * OA_BATCH, len(dois))} / {len(dois)} DOIs")

    oa_map = oa.batch_lookup_dois(dois, on_batch=_on_batch) if dois else {}

    rows = []
    for it in matched:
        doi = it["DOI"]
        w = oa_map.get(doi)
        if w:
            if is_paratext(w):
                continue
            rows.append(oa_to_row(w, base))
        else:
            rows.append({
                **base,
                "doi": doi,
                "title": title_from_cr(it),
                "authors": authors_from_cr(it),
                "abstract": "",
                "keywords": [],
            })

    return rows


# ── Collection: PACM journal ──

def _collect_pacm(venue, year, issn, track, oa, on_step):
    base = {"venue": venue, "year": year}

    on_step("Finding PACM source", None, f"ISSN {issn}")
    src = oa.get_source_by_issn(issn)
    if not src:
        on_step("PACM source not found", 1.0, f"ISSN {issn}")
        return []

    sid = src["id"].split("/")[-1]
    display = src.get("display_name", issn)
    on_step("Listing PACM works", None, display)

    flt = f"primary_location.source.id:{sid},publication_year:{year}"
    rows = []
    count = 0
    for w in oa.list_works(flt):
        count += 1
        if count % 50 == 0:
            on_step("Listing PACM works", None, f"{count} works scanned")
        if is_paratext(w):
            continue
        if track:
            issue = (w.get("biblio") or {}).get("issue") or ""
            if not is_pacm_hci_track(year, issue, track):
                continue
        rows.append(oa_to_row(w, base))

    on_step("PACM collection done", 1.0, f"{len(rows)} papers")
    return rows


# ── Main entry point ──

def collect_papers(venue: str, year: int, stem_doi: str,
                   pacm_issn: str, pacm_track: str,
                   email: str, on_step=None, prefix_cache=None) -> list[dict]:
    """Collect papers from proceedings (Crossref) and/or PACM journal (OpenAlex).

    on_step(label, progress, detail):
        label: str — current step name
        progress: float 0-1 or None for indeterminate
        detail: str — extra info

    prefix_cache: optional dict[(prefix, year) -> list] — share Crossref scan
        results across multiple collect_papers calls in the same run.
    """
    def _step(label, progress=None, detail=""):
        if on_step:
            on_step(label, progress, detail)

    oa = OpenAlex(email)
    all_rows = []

    # Proceedings via Crossref prefix scan
    if stem_doi and stem_doi.strip():
        _step("Proceedings collection", None, stem_doi)
        proc_rows = _collect_proceedings(venue, year, stem_doi.strip(), email, oa, _step,
                                         prefix_cache=prefix_cache)
        all_rows.extend(proc_rows)
        _step("Proceedings done", 1.0, f"{len(proc_rows)} papers")

    # PACM journal collection
    if pacm_issn and pacm_issn.strip():
        _step("PACM collection", None, f"{pacm_issn} track={pacm_track or 'all'}")
        pacm_rows = _collect_pacm(venue, year, pacm_issn.strip(),
                                   pacm_track.strip() if pacm_track else None,
                                   oa, _step)
        # Deduplicate against proceedings rows by DOI
        existing_dois = {r["doi"].lower() for r in all_rows if r.get("doi")}
        new_pacm = [r for r in pacm_rows if r.get("doi", "").lower() not in existing_dois]
        all_rows.extend(new_pacm)
        _step("PACM done", 1.0,
              f"{len(new_pacm)} new papers ({len(pacm_rows) - len(new_pacm)} duplicates)")

    if not all_rows:
        _step("No papers found", 1.0, "")

    _step("Done", 1.0, f"{len(all_rows)} papers total")
    return all_rows
