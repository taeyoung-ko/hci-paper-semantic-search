"""BibTeX export with Crossref content negotiation + fallback.

Used by ui.py. Lives in its own module so ui.py syntax can't be broken
by escaping issues in helper code.
"""

import datetime
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _fetch_crossref_bibtex(doi, timeout=5):
    """Fetch a BibTeX entry from doi.org content negotiation. Returns the
    raw BibTeX text on success, or None on any failure."""
    if not doi:
        return None
    url = "https://doi.org/" + doi
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/x-bibtex; charset=utf-8",
            "User-Agent": "HCIPaperSearch/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        text = data.strip()
        if text.startswith("@"):
            return text
        print(f"[bib] non-bibtex response for {doi}: {text[:80]!r}", flush=True)
        return None
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError) as e:
        print(f"[bib] crossref fetch failed for {doi}: {e}", flush=True)
        return None


def _latex_escape(text):
    """Minimal LaTeX-safe escaping for plain text fields."""
    if not text:
        return ""
    s = str(text)
    # Order matters: backslash first
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def _make_cite_key(entry, used_keys):
    """Build a unique BibTeX cite key from author/year/title."""
    authors = entry.get("authors") or []
    first_author = authors[0] if authors else ""
    surname = first_author.split()[-1] if first_author else "anon"
    surname = re.sub(r"[^A-Za-z]", "", surname).lower() or "anon"

    year = str(entry.get("year") or "")
    year_part = re.sub(r"[^0-9]", "", year)[:4] or "nodate"

    title = entry.get("title") or ""
    stop = {"the", "a", "an", "of", "on", "in", "and", "for", "to", "with"}
    word = ""
    for tok in re.split(r"[^A-Za-z]+", title):
        if tok and tok.lower() not in stop:
            word = tok.lower()
            break
    word = word or "paper"

    base = f"{surname}{year_part}{word}"
    key = base
    suffix = 0
    while key in used_keys:
        suffix += 1
        key = base + chr(ord("a") + suffix - 1)
    used_keys.add(key)
    return key


def _fallback_bibtex(entry, cite_key):
    """Build a BibTeX entry from our own metadata when Crossref fails."""
    authors = entry.get("authors") or []
    if authors:
        formatted = []
        for a in authors:
            parts = a.strip().split()
            if len(parts) >= 2:
                formatted.append(parts[-1] + ", " + " ".join(parts[:-1]))
            else:
                formatted.append(a.strip())
        authors_str = " and ".join(formatted)
    else:
        authors_str = ""

    venue = entry.get("venue") or ""
    year = entry.get("year") or ""

    venue_to_booktitle = {
        "CHI":    f"Proceedings of the {year} CHI Conference on Human Factors in Computing Systems",
        "UIST":   f"Proceedings of the {year} ACM Symposium on User Interface Software and Technology",
        "CSCW":   "Proceedings of the ACM on Human-Computer Interaction",
        "DIS":    f"Proceedings of the {year} ACM Designing Interactive Systems Conference",
        "IUI":    f"Proceedings of the {year} ACM Conference on Intelligent User Interfaces",
        "RecSys": f"Proceedings of the {year} ACM Conference on Recommender Systems",
        "UMAP":   f"Proceedings of the {year} ACM Conference on User Modeling, Adaptation and Personalization",
    }
    booktitle = venue_to_booktitle.get(venue, f"Proceedings of {venue} {year}".strip())

    doi = entry.get("doi") or ""
    title = entry.get("title") or ""
    abstract = entry.get("abstract") or ""
    fields = []
    if authors_str:
        fields.append(f"  author    = {{{_latex_escape(authors_str)}}},")
    if title:
        fields.append(f"  title     = {{{_latex_escape(title)}}},")
    if booktitle:
        fields.append(f"  booktitle = {{{_latex_escape(booktitle)}}},")
    if year:
        fields.append(f"  year      = {{{year}}},")
    fields.append("  publisher = {Association for Computing Machinery},")
    fields.append("  address   = {New York, NY, USA},")
    if doi:
        fields.append(f"  doi       = {{{doi}}},")
        fields.append(f"  url       = {{https://doi.org/{doi}}},")
    if abstract:
        fields.append(f"  abstract  = {{{_latex_escape(abstract)}}},")

    body = "\n".join(fields)
    return f"@inproceedings{{{cite_key},\n{body}\n}}"


def _replace_cite_key(bibtex_text, new_key):
    """Replace the cite key in a BibTeX entry (between { and ,)."""
    return re.sub(
        r"^(@\w+\{)[^,]+",
        lambda m: m.group(1) + new_key,
        bibtex_text,
        count=1,
    )


def _metadata_comments(entry):
    """Build the % ... comment lines for an entry."""
    src = entry.get("source") or {}
    note = entry.get("user_note") or ""
    lines = []
    mode_label = src.get("mode_label") or src.get("mode") or ""
    if mode_label:
        lines.append(f"% Found in: {mode_label} search")
    q = src.get("query") or ""
    if q:
        q1 = q.replace("\n", " ").replace("\r", " ")
        lines.append(f"% Query: {q1}")
    score = src.get("rerank_score")
    if score:
        lines.append(f"% Score: {float(score):.3f}")
    added = src.get("added_at") or ""
    if added:
        lines.append(f"% Added: {added}")
    if note:
        for ln in note.splitlines() or [note]:
            lines.append(f"% Note: {ln}")
    return "\n".join(lines)


def export_bibtex(collection):
    """Build the full .bib file content for the current collection.
    Returns (content, count) where count is number of entries written.
    """
    if not collection:
        header = ("% Bibliography exported from HCI Paper Semantic Search\n"
                  "% (collection is empty)\n")
        return header, 0

    # Generate BibTeX entries directly from our metadata (no Crossref).
    # We have richer abstract/keywords than Crossref typically provides, and
    # our format is consistent across all entries.
    print(f"[bib] generating {len(collection)} entries from local metadata...", flush=True)

    used_keys = set()
    chunks = []
    plural = "s" if len(collection) != 1 else ""
    chunks.append("% Bibliography exported from HCI Paper Semantic Search")
    chunks.append(f"% {len(collection)} reference{plural}")
    chunks.append("")

    for entry in collection:
        cite_key = _make_cite_key(entry, used_keys)
        bib = _fallback_bibtex(entry, cite_key)
        meta = _metadata_comments(entry)
        chunks.append(bib)
        if meta:
            chunks.append(meta)
        chunks.append("")

    print(f"[bib] export done. {len(collection)} entries written.", flush=True)
    return "\n".join(chunks), len(collection)


def write_bib_to_tempfile(collection):
    """Write the .bib content to a temp file and return its path. Used by
    Gradio gr.File output."""
    content, n = export_bibtex(collection or [])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"hcips_collection_{ts}.bib"

    fp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".bib", delete=False, prefix="hcips_"
    )
    fp.write(content)
    fp.close()

    target = os.path.join(os.path.dirname(fp.name), fname)
    shutil.move(fp.name, target)
    print(f"[bib] wrote {target} ({n} entries)", flush=True)
    return target
