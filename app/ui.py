# Monkey-patch gradio_client/utils.py to handle bool schema values
# (Gradio 4.x bug: get_type() crashes on bool schema, e.g. additionalProperties=True)
import gradio_client.utils as _gc_utils

_gc_utils_get_type_orig = _gc_utils.get_type

def _gc_utils_get_type_safe(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _gc_utils_get_type_orig(schema)

_gc_utils.get_type = _gc_utils_get_type_safe

_gc_utils_jsts_orig = _gc_utils._json_schema_to_python_type

def _gc_utils_jsts_safe(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _gc_utils_jsts_orig(schema, defs)

_gc_utils._json_schema_to_python_type = _gc_utils_jsts_safe

import html
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
import secrets
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import plotly.io as pio
import umap

from search import search, get_filter_options
from bib_export import write_bib_to_tempfile

_OPTS = get_filter_options()
_VENUES = _OPTS["venues"]


# ──────────────────────── Background warmup ────────────────────────

_warmup_done = threading.Event()


def _warmup():
    print("[ui] Warming up UMAP/numba JIT in background (~30s)...")
    _dummy = np.random.randn(30, 32).astype("float32")
    umap.UMAP(n_components=2, n_neighbors=5, random_state=42).fit_transform(_dummy)
    _warmup_done.set()
    print("[ui] Warmup complete.")


threading.Thread(target=_warmup, daemon=True).start()


# ──────────────────────── Constants ────────────────────────

# UI label -> backend mode key (must match MODES in api_clients.py)
COMPONENT_BOXES = [
    ("Background", "background"),
    ("Gap", "gap"),
    ("Solution", "solution"),
    ("Approach/Method", "method"),
    ("Findings", "findings"),
]

MAX_TABS = 6  # Up to 1 Overall + 5 component results


# ──────────────────────── Helpers ────────────────────────

def _score_to_color(score):
    s = max(0.0, min(1.0, float(score)))
    hue = int(120 * s)
    return f"hsl({hue}, 75%, 55%)", "#ffffff"


def _truncate(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n].rstrip() + "..."


# ──────────────────────── List view ────────────────────────

def _render_card(rank, p):
    title = html.escape(p.get("title") or "(no title)")
    abstract = html.escape(p.get("abstract") or "")
    venue = html.escape(str(p.get("venue") or ""))
    year = html.escape(str(p.get("year") or ""))
    doi = p.get("doi") or ""
    score = float(p.get("rerank_score", 0.0))
    bg, fg = _score_to_color(score)

    authors = p.get("authors") or []
    if len(authors) > 4:
        authors_str = ", ".join(authors[:4]) + f", and {len(authors) - 4} more"
    else:
        authors_str = ", ".join(authors)
    authors_str = html.escape(authors_str) if authors_str else "(no authors)"

    keywords = p.get("keywords") or []
    keywords_html = ""
    if keywords:
        chips_parts = []
        for k in keywords:
            chips_parts.append(
                '<span style="display:inline-block;padding:2px 8px;'
                'margin:2px 4px 2px 0;background:#eef2f7;color:#374151;'
                'border-radius:10px;font-size:0.82em;">'
                + html.escape(k) + '</span>'
            )
        chips = "".join(chips_parts)
        keywords_html = '<div style="margin:8px 0 4px 0;">' + chips + '</div>'

    abstract_html = ""
    if abstract:
        abstract_html = (
            '<div style="font-size:0.93em;color:#333;line-height:1.55;'
            'margin-top:6px;white-space:pre-wrap;">' + abstract + '</div>'
        )

    if doi:
        title_link = (
            '<a href="https://doi.org/' + html.escape(doi) + '" target="_blank" '
            'style="font-weight:600;font-size:1.05em;color:#1d4ed8;'
            'text-decoration:none;">' + title + '</a>'
        )
    else:
        title_link = (
            '<span style="font-weight:600;font-size:1.05em;color:#111;">'
            + title + '</span>'
        )

    score_badge = (
        '<span style="display:inline-block;padding:4px 10px;border-radius:8px;'
        'background:' + bg + ';color:' + fg + ';font-weight:700;'
        'font-size:0.92em;min-width:60px;text-align:center;">'
        + f"{score:.3f}" + '</span>'
    )

    if venue or year:
        meta = (
            '<span style="color:#6b7280;">#' + str(rank) + ' &middot; '
            + venue + ' ' + year + '</span>'
        )
    else:
        meta = '<span style="color:#6b7280;">#' + str(rank) + '</span>'

    return (
        '<div style="border-bottom:1px solid #e5e7eb;padding:16px 6px;">'
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">'
        + score_badge + '<div style="font-size:0.88em;">' + meta + '</div></div>'
        '<div>' + title_link + '</div>'
        '<div style="font-size:0.9em;color:#4b5563;margin-top:4px;">'
        + authors_str + '</div>'
        + keywords_html + abstract_html + '</div>'
    )


def _render_list(results, search_context=None):
    if not results:
        return "<div style='color:#6b7280;padding:12px;'>No results.</div>"
    cards_html = "\n".join(_render_card(i + 1, p) for i, p in enumerate(results))
    return _wrap_list_in_iframe(cards_html, results, search_context=search_context)


def _wrap_list_in_iframe(cards_html, results, search_context=None):
    """Wrap rendered card HTML in an iframe so we can attach interactive
    star buttons. For now the buttons are visual-only placeholders; in a
    later step they will postMessage to the parent on click."""

    # Build a JS-side dictionary of paper metadata keyed by DOI so the iframe
    # can include it in the postMessage payload when a star is clicked.
    import json as _json
    meta_for_js = {}
    for p in results:
        doi = p.get("doi") or p.get("title") or ""
        if not doi:
            continue
        meta_for_js[doi] = {
            "doi": doi,
            "title": p.get("title") or "",
            "authors": p.get("authors") or [],
            "venue": p.get("venue") or "",
            "year": p.get("year") or "",
            "abstract": p.get("abstract") or "",
            "keywords": p.get("keywords") or [],
            "rerank_score": float(p.get("rerank_score", 0.0)),
        }
    meta_json = _json.dumps(meta_for_js)

    star_css = """
    <style>
      html, body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; }
      .star-btn {
        position: absolute; top: 14px; right: 10px;
        background: transparent; border: none; cursor: pointer;
        font-size: 22px; line-height: 1; padding: 4px 6px;
        color: #9ca3af; transition: color 120ms, transform 120ms;
      }
      .star-btn:hover { color: #f59e0b; transform: scale(1.15); }
      .star-btn.active { color: #f59e0b; }
      .card-wrap { position: relative; }
    </style>
    """

    star_js = """
    <script>
      window.__paperMeta = %s;
      window.__searchContext = %s;
      // Decorate each card with a star button
      document.addEventListener('DOMContentLoaded', function() {
        var cards = document.querySelectorAll('[data-doi]');
        cards.forEach(function(card) {
          var doi = card.getAttribute('data-doi');
          if (!doi) return;
          var btn = document.createElement('button');
          btn.className = 'star-btn';
          btn.setAttribute('aria-label', 'Add to collection');
          btn.setAttribute('data-doi', doi);
          btn.innerText = '\u2606';  // hollow star
          btn.addEventListener('click', function(e) {
            e.preventDefault();
            var nowActive = !this.classList.contains('active');
            this.classList.toggle('active', nowActive);
            this.innerText = nowActive ? '\u2605' : '\u2606';
            // Send the toggle event up to the parent Gradio app
            try {
              var meta = (window.__paperMeta && window.__paperMeta[doi]) || null;
              window.parent.postMessage({
                __hcips: true,
                type: 'toggleStar',
                doi: doi,
                active: nowActive,
                meta: meta,
                searchContext: window.__searchContext || null,
              }, '*');
            } catch (err) { console.error(err); }
          });
          // Reflect already-collected state on render
          try {
            var collected = (window.parent && window.parent.__hcipsCollectedDois) || [];
            if (collected.indexOf(doi) >= 0) {
              btn.classList.add('active');
              btn.innerText = '\u2605';
            }
          } catch (err) {}
          card.classList.add('card-wrap');
          card.insertBefore(btn, card.firstChild);
        });
        // Auto-resize iframe to its content
        function fit() {
          var h = document.documentElement.scrollHeight;
          if (window.parent && window.frameElement) {
            window.frameElement.style.height = h + 'px';
          }
        }
        fit();
        new ResizeObserver(fit).observe(document.body);

        // Listen for "syncStars" broadcasts from the parent so this iframe's
        // star buttons can re-render to match the current collection.
        window.addEventListener('message', function(ev) {
          var d = ev.data;
          if (!d || d.__hcipsSync !== true) return;
          var dois = (d.dois || []);
          var stars = document.querySelectorAll('.star-btn');
          stars.forEach(function(s) {
            var doi = s.getAttribute('data-doi');
            var on = doi && dois.indexOf(doi) >= 0;
            s.classList.toggle('active', on);
            s.innerText = on ? '\u2605' : '\u2606';
          });
        });
      });
    </script>
    """ % (meta_json, _json.dumps(search_context or {}))

    # We need each card's outer <div> to carry a data-doi attribute so the
    # script can find it. _render_card doesn't add that today, so wrap each
    # card in a container that does.
    # Instead of editing _render_card, do a light transform here: split on
    # the known opening div and inject data-doi from the results list.
    decorated_cards = []
    for i, p in enumerate(results):
        doi = p.get("doi") or p.get("title") or ""
        card_html = _render_card(i + 1, p)
        # Add data-doi to the outer div by replacing its opening tag once
        opener = '<div style="border-bottom:1px solid #e5e7eb;padding:16px 6px;">'
        opener_with_doi = (
            '<div data-doi="' + html.escape(doi, quote=True) + '" '
            'style="border-bottom:1px solid #e5e7eb;padding:16px 6px;">'
        )
        if opener in card_html:
            card_html = card_html.replace(opener, opener_with_doi, 1)
        decorated_cards.append(card_html)
    cards_html_decorated = "\n".join(decorated_cards)

    full_html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        + star_css
        + '</head><body>'
        + cards_html_decorated
        + star_js
        + '</body></html>'
    )

    escaped = html.escape(full_html, quote=True)
    return (
        '<iframe srcdoc="' + escaped + '" '
        'style="display:block;width:100%;border:0;min-height:200px;"></iframe>'
    )


# ──────────────────────── Graph view ────────────────────────

_EMPTY_GRAPH_HTML = (
    "<div style='color:#6b7280;padding:32px;text-align:center;'>"
    "Run a search to see the graph.</div>"
)


def _build_graph_html(query, query_emb, results):
    n = len(results)
    if n == 0 or query_emb is None:
        return ("<div style='color:#6b7280;padding:32px;text-align:center;'>"
                "No results.</div>")

    vecs = np.vstack(
        [query_emb.reshape(-1)] + [p["_vec"] for p in results]
    ).astype("float32")

    n_neighbors = min(15, max(2, n - 1))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        metric="cosine", min_dist=0.1, random_state=42,
    )
    coords = reducer.fit_transform(vecs)
    q_xy = coords[0]
    paper_xy = coords[1:] - q_xy

    radii = np.array([1.0 - float(p.get("rerank_score", 0.0)) for p in results])
    norms = np.linalg.norm(paper_xy, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    final_xy = (paper_xy / norms) * radii.reshape(-1, 1)

    scores = np.array([float(p.get("rerank_score", 0.0)) for p in results])
    sizes = 10 + scores * 30

    hover_texts = []
    customdata = []
    for p in results:
        title = html.escape(p.get("title") or "(no title)")
        authors_list = p.get("authors") or []
        if authors_list:
            authors_str = html.escape(", ".join(authors_list))
        else:
            authors_str = "(no authors)"
        venue = html.escape(str(p.get("venue") or ""))
        year = html.escape(str(p.get("year") or ""))
        score = float(p.get("rerank_score", 0.0))
        hover_texts.append(
            "<b>" + title + "</b><br>"
            + authors_str + "<br>"
            + venue + " " + year + "<br>"
            + f"score {score:.3f}"
        )
        doi = p.get("doi") or ""
        customdata.append("https://doi.org/" + doi if doi else "")

    fig = go.Figure()

    edge_x = []
    edge_y = []
    for (x, y) in final_xy:
        edge_x.extend([0.0, float(x), None])
        edge_y.extend([0.0, float(y), None])
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(120,120,120,0.25)", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=final_xy[:, 0], y=final_xy[:, 1], mode="markers",
        marker=dict(
            size=sizes, color=scores,
            colorscale=[
                [0.0, "hsl(0, 75%, 55%)"],
                [0.5, "hsl(60, 75%, 55%)"],
                [1.0, "hsl(120, 75%, 55%)"],
            ],
            cmin=0.0, cmax=1.0,
            line=dict(color="#ffffff", width=1.5),
            colorbar=dict(title="rerank<br>score", thickness=14, len=0.6),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        customdata=customdata,
        showlegend=False, name="papers",
    ))

    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(
            size=24, color="#111", symbol="star",
            line=dict(color="#fff", width=2),
        ),
        text=["Query"], textposition="top center",
        textfont=dict(size=12, color="#111"),
        hovertext=_truncate(query, 200), hoverinfo="text",
        showlegend=False,
    ))

    if final_xy.size > 0:
        max_r = float(np.max(np.abs(final_xy))) * 1.10
    else:
        max_r = 1.0
    if max_r < 1e-6:
        max_r = 1.0

    fig.update_layout(
        template="simple_white", autosize=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-max_r, max_r], constrain="domain"),
        yaxis=dict(visible=False, range=[-max_r, max_r], constrain="domain"),
    )

    div_id = "umap_graph_" + secrets.token_hex(4)
    plot_div = pio.to_html(
        fig,
        include_plotlyjs=True, full_html=False, div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
        default_height="100%", default_width="100%",
    )

    click_js = (
        "<script>(function(){"
        "function attach(){"
        "var el=document.getElementById('" + div_id + "');"
        "if(!el||!el.on){return setTimeout(attach,100);}"
        "el.on('plotly_click',function(data){"
        "if(!data||!data.points||data.points.length===0)return;"
        "var url=data.points[0].customdata;"
        "if(url){window.open(url,'_blank');}"
        "});"
        "}"
        "attach();"
        "})();</script>"
    )

    self_resize_js = (
        "<script>(function(){"
        "function fit(){try{"
        "var fe=window.parent&&window.frameElement;"
        "if(!fe)return;"
        "var w=fe.getBoundingClientRect().width;"
        "if(w<=0)return;"
        "fe.style.height=w+'px';"
        "if(window.Plotly){"
        "var divs=document.querySelectorAll('.plotly-graph-div');"
        "divs.forEach(function(d){try{"
        "window.Plotly.relayout(d,{width:w,height:w,autosize:false});"
        "}catch(e){}});"
        "}"
        "}catch(e){}}"
        "fit();"
        "setTimeout(fit,50);setTimeout(fit,200);"
        "setTimeout(fit,600);setTimeout(fit,1500);"
        "window.addEventListener('resize',fit);"
        "try{var ro=new ResizeObserver(fit);"
        "if(window.frameElement&&window.frameElement.parentElement){"
        "ro.observe(window.frameElement.parentElement);"
        "}}catch(e){}"
        "})();</script>"
    )

    head_style = (
        "<style>"
        "html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;}"
        ".plotly-graph-div{width:100%!important;height:100%!important;}"
        ".svg-container{width:100%!important;height:100%!important;}"
        "</style>"
    )

    full_html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        + head_style
        + '</head><body style="margin:0;width:100vw;height:100vh;overflow:hidden;">'
        + plot_div + click_js + self_resize_js
        + '</body></html>'
    )

    escaped = html.escape(full_html, quote=True)
    return (
        '<iframe srcdoc="' + escaped + '" '
        'style="display:block;width:100%;height:600px;border:0;"></iframe>'
    )


# ──────────────────────── Combined (RRF) ────────────────────────

RRF_K = 60


def _combine_rrf(per_mode_results, rerank_k):
    """Reciprocal Rank Fusion over the rankings of multiple modes.

    per_mode_results: list of (mode_label, list_of_paper_dicts)
        each paper_dict already has rerank_score and metadata from search().
    Returns: list of dicts, sorted by RRF score, length <= rerank_k.
    Each dict carries:
      - all metadata of the paper (title, authors, doi, abstract, etc.)
      - "_rrf_score": float
      - "_rrf_modes": dict {mode_label: rank (1-based)} only for modes the paper
        appeared in.
    """
    by_doi = {}  # doi -> aggregate dict
    for (mode_label, results) in per_mode_results:
        for rank_idx, p in enumerate(results, start=1):
            doi = p.get("doi") or p.get("title")  # fallback dedupe key
            if not doi:
                continue
            entry = by_doi.get(doi)
            if entry is None:
                entry = dict(p)  # copy metadata
                entry["_rrf_score"] = 0.0
                entry["_rrf_modes"] = {}
                by_doi[doi] = entry
            entry["_rrf_score"] += 1.0 / (RRF_K + rank_idx)
            entry["_rrf_modes"][mode_label] = rank_idx

    fused = sorted(by_doi.values(), key=lambda d: d["_rrf_score"], reverse=True)
    return fused[:max(1, int(rerank_k))]


def _render_card_combined(rank, p):
    """Variant of _render_card that shows RRF score + per-mode rank chips."""
    title = html.escape(p.get("title") or "(no title)")
    abstract = html.escape(p.get("abstract") or "")
    venue = html.escape(str(p.get("venue") or ""))
    year = html.escape(str(p.get("year") or ""))
    doi = p.get("doi") or ""

    rrf_score = float(p.get("_rrf_score", 0.0))
    # Map RRF score onto a 0..1 visual scale. Max possible (5 modes all rank 1)
    # is 5/(60+1) ≈ 0.082. Use that as the upper anchor for color.
    visual = max(0.0, min(1.0, rrf_score / 0.082))
    bg, fg = _score_to_color(visual)

    authors = p.get("authors") or []
    if len(authors) > 4:
        authors_str = ", ".join(authors[:4]) + f", and {len(authors) - 4} more"
    else:
        authors_str = ", ".join(authors)
    authors_str = html.escape(authors_str) if authors_str else "(no authors)"

    keywords = p.get("keywords") or []
    keywords_html = ""
    if keywords:
        chips_parts = []
        for k in keywords:
            chips_parts.append(
                '<span style="display:inline-block;padding:2px 8px;'
                'margin:2px 4px 2px 0;background:#eef2f7;color:#374151;'
                'border-radius:10px;font-size:0.82em;">'
                + html.escape(k) + '</span>'
            )
        keywords_html = '<div style="margin:8px 0 4px 0;">' + "".join(chips_parts) + '</div>'

    # Per-mode rank chips
    mode_chips_html = ""
    modes = p.get("_rrf_modes") or {}
    if modes:
        chip_parts = []
        for mode_label, mode_rank in modes.items():
            chip_parts.append(
                '<span style="display:inline-block;padding:2px 8px;'
                'margin:2px 4px 2px 0;background:#dbeafe;color:#1e3a8a;'
                'border-radius:10px;font-size:0.82em;font-weight:600;">'
                + html.escape(mode_label) + " #" + str(mode_rank) + '</span>'
            )
        mode_chips_html = '<div style="margin:6px 0;">' + "".join(chip_parts) + '</div>'

    abstract_html = ""
    if abstract:
        abstract_html = (
            '<div style="font-size:0.93em;color:#333;line-height:1.55;'
            'margin-top:6px;white-space:pre-wrap;">' + abstract + '</div>'
        )

    if doi:
        title_link = (
            '<a href="https://doi.org/' + html.escape(doi) + '" target="_blank" '
            'style="font-weight:600;font-size:1.05em;color:#1d4ed8;'
            'text-decoration:none;">' + title + '</a>'
        )
    else:
        title_link = (
            '<span style="font-weight:600;font-size:1.05em;color:#111;">'
            + title + '</span>'
        )

    score_badge = (
        '<span style="display:inline-block;padding:4px 10px;border-radius:8px;'
        'background:' + bg + ';color:' + fg + ';font-weight:700;'
        'font-size:0.92em;min-width:60px;text-align:center;">'
        + f"{rrf_score:.4f}" + '</span>'
    )

    if venue or year:
        meta = (
            '<span style="color:#6b7280;">#' + str(rank) + ' &middot; '
            + venue + ' ' + year + '</span>'
        )
    else:
        meta = '<span style="color:#6b7280;">#' + str(rank) + '</span>'

    return (
        '<div style="border-bottom:1px solid #e5e7eb;padding:16px 6px;">'
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">'
        + score_badge + '<div style="font-size:0.88em;">' + meta + '</div></div>'
        '<div>' + title_link + '</div>'
        '<div style="font-size:0.9em;color:#4b5563;margin-top:4px;">'
        + authors_str + '</div>'
        + mode_chips_html + keywords_html + abstract_html + '</div>'
    )


def _render_list_combined(results):
    if not results:
        return "<div style='color:#6b7280;padding:12px;'>No results.</div>"
    return "\n".join(_render_card_combined(i + 1, p) for i, p in enumerate(results))




# ──────────────────────── Collection (starred papers) ────────────────────────

import datetime as _dt


def _save_note(collection, payload_json):
    """Backend handler for saveNote messages. Updates user_note for the
    matching DOI without touching the rest of the collection."""
    import json as _json
    print(f"[note] handler invoked. payload_len={len(payload_json or '')}", flush=True)

    if collection is None:
        collection = []
    if not payload_json:
        return collection
    try:
        payload = _json.loads(payload_json)
    except Exception as e:
        print(f"[note] bad payload: {e}", flush=True)
        return collection

    doi = (payload.get("doi") or "").strip()
    if not doi:
        return collection
    note = payload.get("note") or ""
    for entry in collection:
        if entry.get("doi") == doi:
            entry["user_note"] = note
            print(f"[note] saved for {doi} ({len(note)} chars)", flush=True)
            break
    return collection


def _toggle_collection(collection, payload_json):
    """Backend handler for star toggle (add/remove papers from collection)."""
    import json as _json
    import sys
    print(f"[collection] handler invoked. payload_len={len(payload_json or '')}", flush=True)

    if collection is None:
        collection = []
    if not payload_json:
        print("[collection] empty payload, no-op", flush=True)
        return collection
    try:
        payload = _json.loads(payload_json)
    except Exception as e:
        print(f"[collection] bad payload: {e}", flush=True)
        return collection

    doi = (payload.get("doi") or "").strip()
    if not doi:
        return collection

    active = bool(payload.get("active"))
    meta = payload.get("meta") or {}
    sctx = payload.get("searchContext") or {}

    existing_idx = next(
        (i for i, e in enumerate(collection) if e.get("doi") == doi),
        None,
    )

    if active:
        if existing_idx is not None:
            return collection  # already starred, no-op
        entry = {
            "doi": doi,
            "title": meta.get("title", ""),
            "authors": meta.get("authors", []),
            "venue": meta.get("venue", ""),
            "year": meta.get("year", ""),
            "abstract": meta.get("abstract", ""),
            "keywords": meta.get("keywords", []),
            "source": {
                "mode": sctx.get("mode", ""),
                "mode_label": sctx.get("mode_label", ""),
                "query": sctx.get("query", ""),
                "rerank_score": meta.get("rerank_score", 0.0),
                "added_at": _dt.datetime.now().isoformat(timespec="seconds"),
            },
            "user_note": "",
        }
        collection.append(entry)
        print(f"[collection] +{doi} ({len(collection)} total)", flush=True)
    else:
        if existing_idx is not None:
            collection.pop(existing_idx)
            print(f"[collection] -{doi} ({len(collection)} total)", flush=True)

    return collection


# ──────────────────────── Collection rendering ────────────────────────

def _render_collection(collection):
    """Render the starred-paper collection as an iframe with remove buttons.

    Each card has:
      - title (DOI link)
      - authors, venue, year
      - source toggle (chevron) that expands to show how/why it was added
      - remove button [x]
    Also emits a script tag that broadcasts the current DOI list to all
    iframes (to keep their star buttons in sync).
    """
    import json as _json
    dois_for_sync = [e.get("doi") for e in (collection or []) if e.get("doi")]
    sync_script = (
        "<script>(function(){"
        "var dois = " + _json.dumps(dois_for_sync) + ";"
        "window.__hcipsCollectedDois = dois;"
        "var iframes = window.parent.document.querySelectorAll('iframe');"
        "iframes.forEach(function(f){"
        "try{f.contentWindow.postMessage({__hcipsSync:true,dois:dois},'*');}"
        "catch(e){}"
        "});"
        "})();</script>"
    )
    if not collection:
        return (
            sync_script
            + "<div style='color:#6b7280;padding:18px;text-align:center;"
            "font-size:0.95em;'>"
            "Click the &#9734; star on any search result to add it here."
            "</div>"
        )

    cards = []
    for entry in collection:
        doi = entry.get("doi") or ""
        title = html.escape(entry.get("title") or "(no title)")
        authors_list = entry.get("authors") or []
        if len(authors_list) > 4:
            authors_str = ", ".join(authors_list[:4]) + f", and {len(authors_list)-4} more"
        else:
            authors_str = ", ".join(authors_list) if authors_list else "(no authors)"
        authors_str = html.escape(authors_str)
        venue = html.escape(str(entry.get("venue") or ""))
        year = html.escape(str(entry.get("year") or ""))

        src = entry.get("source") or {}
        src_mode_label = html.escape(src.get("mode_label") or src.get("mode") or "")
        src_query = html.escape(src.get("query") or "")
        src_score = src.get("rerank_score") or 0.0
        src_added = html.escape(src.get("added_at") or "")

        title_link = (
            f'<a href="https://doi.org/{html.escape(doi)}" target="_blank" '
            f'style="font-weight:600;color:#1d4ed8;text-decoration:none;">{title}</a>'
            if doi else
            f'<span style="font-weight:600;color:#111;">{title}</span>'
        )

        meta_line = ""
        if venue or year:
            meta_line = (
                f'<div style="color:#6b7280;font-size:0.88em;margin-top:2px;">'
                f'{venue} {year}</div>'
            )

        # Collapsible source detail
        src_detail = ""
        if src_mode_label or src_query:
            src_detail = (
                '<details style="margin-top:8px;font-size:0.85em;color:#4b5563;">'
                '<summary style="cursor:pointer;color:#6b7280;'
                'font-weight:500;">source</summary>'
                '<div style="margin:6px 0 0 4px;line-height:1.5;">'
                + (f'<div><b>Found in:</b> {src_mode_label} search</div>' if src_mode_label else '')
                + (f'<div><b>Query:</b> {src_query}</div>' if src_query else '')
                + (f'<div><b>Score:</b> {src_score:.3f}</div>' if src_score else '')
                + (f'<div><b>Added:</b> {src_added}</div>' if src_added else '')
                + '</div></details>'
            )

        remove_btn = (
            f'<button class="rm-btn" data-doi="{html.escape(doi, quote=True)}" '
            f'aria-label="Remove from collection" '
            f'style="position:absolute;top:10px;right:10px;background:transparent;'
            f'border:none;cursor:pointer;font-size:18px;color:#9ca3af;'
            f'padding:4px 6px;line-height:1;">&times;</button>'
        )

        # Note UI: "Add note" link if empty, expanded textarea if has note,
        # toggleable in either case.
        existing_note = entry.get("user_note") or ""
        note_html = (
            '<div class="note-block" data-doi="' + html.escape(doi, quote=True) + '" '
            'style="margin-top:8px;font-size:0.88em;">'
            '<a href="javascript:void(0)" class="note-toggle" '
            'style="color:#6b7280;text-decoration:none;cursor:pointer;'
            'border-bottom:1px dashed #d1d5db;">'
            + ('Edit note' if existing_note else '+ Add note')
            + '</a>'
            '<div class="note-editor" style="display:'
            + ('block' if existing_note else 'none')
            + ';margin-top:6px;">'
            '<textarea class="note-input" '
            'data-doi="' + html.escape(doi, quote=True) + '" '
            'placeholder="Add a note..." '
            'style="width:100%;min-height:24px;resize:none;overflow:hidden;'
            'box-sizing:border-box;padding:6px 8px;'
            'border:1px solid #d1d5db;border-radius:6px;'
            'font-family:inherit;font-size:0.92em;line-height:1.4;'
            'background:#fff;">'
            + html.escape(existing_note) + '</textarea>'
            '</div></div>'
        )

        cards.append(
            '<div style="position:relative;border:1px solid #e5e7eb;'
            'border-radius:8px;padding:14px 36px 14px 14px;margin-bottom:10px;'
            'background:#fafafa;">'
            + remove_btn
            + '<div>' + title_link + '</div>'
            + '<div style="font-size:0.9em;color:#4b5563;margin-top:4px;">'
            + authors_str + '</div>'
            + meta_line
            + note_html
            + src_detail
            + '</div>'
        )

    cards_html = "\n".join(cards)

    js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
      var btns = document.querySelectorAll('.rm-btn');
      btns.forEach(function(btn) {
        btn.addEventListener('mouseenter', function() { this.style.color = '#dc2626'; });
        btn.addEventListener('mouseleave', function() { this.style.color = '#9ca3af'; });
        btn.addEventListener('click', function(e) {
          e.preventDefault();
          var doi = this.getAttribute('data-doi');
          if (!doi) return;
          try {
            window.parent.postMessage({
              __hcips: true,
              type: 'toggleStar',
              doi: doi,
              active: false,
              meta: null,
              searchContext: null,
            }, '*');
          } catch (err) { console.error(err); }
        });
      });

      // "Add note" / "Edit note" toggle
      var toggles = document.querySelectorAll('.note-toggle');
      toggles.forEach(function(tog) {
        tog.addEventListener('click', function(e) {
          e.preventDefault();
          var block = this.closest('.note-block');
          if (!block) return;
          var editor = block.querySelector('.note-editor');
          if (!editor) return;
          var open = editor.style.display !== 'none';
          editor.style.display = open ? 'none' : 'block';
          if (!open) {
            var ta = editor.querySelector('.note-input');
            if (ta) {
              ta.focus();
              autoResize(ta);
            }
          }
        });
      });

      // Auto-resize textarea + onblur autosave
      function autoResize(ta) {
        ta.style.height = 'auto';
        ta.style.height = (ta.scrollHeight) + 'px';
      }
      var notes = document.querySelectorAll('.note-input');
      notes.forEach(function(ta) {
        autoResize(ta);
        ta.addEventListener('input', function() { autoResize(this); });
        ta.addEventListener('blur', function() {
          var doi = this.getAttribute('data-doi');
          var text = this.value;
          if (!doi) return;
          try {
            window.parent.postMessage({
              __hcips: true,
              type: 'saveNote',
              doi: doi,
              note: text,
            }, '*');
          } catch (err) { console.error(err); }
        });
      });
      // Auto-resize iframe to its content
      function fit() {
        var h = document.documentElement.scrollHeight;
        if (window.parent && window.frameElement) {
          window.frameElement.style.height = h + 'px';
        }
      }
      fit();
      new ResizeObserver(fit).observe(document.body);
    });
    </script>
    """

    css = """
    <style>
      html, body { margin: 0; padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; }
      .rm-btn:hover { color: #dc2626 !important; }
      details summary::-webkit-details-marker { color: #6b7280; }
    </style>
    """

    full_html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        + css
        + '</head><body>'
        + cards_html
        + js
        + '</body></html>'
    )

    escaped = html.escape(full_html, quote=True)
    return (
        sync_script
        + '<iframe srcdoc="' + escaped + '" '
        'style="display:block;width:100%;border:0;min-height:80px;"></iframe>'
    )


def _collection_label(collection):
    """Return the dynamic label for the Accordion header."""
    n = len(collection) if collection else 0
    if n == 0:
        return "My Collection (empty)"
    return f"My Collection ({n} paper{'s' if n != 1 else ''})"


# ──────────────────────── Per-session search cache ────────────────────────

# We cache the result of `search()` keyed on (mode, query_text, venues_tuple,
# retrieve_k, rerank_k). The cache lives in a gr.State (per-browser-session)
# so that users don't share entries. Each mode keeps at most CACHE_PER_MODE
# entries; oldest entries are evicted (LRU).

CACHE_PER_MODE = 5


def _cache_key(mode, qtext, venues, retrieve_k, rerank_k):
    venues_t = tuple(sorted(venues)) if venues else ()
    return (mode, qtext, venues_t, int(retrieve_k), int(rerank_k))


def _cache_get(cache, key):
    """Return cached result or None. Move entry to end (most-recent)."""
    if cache is None:
        return None
    mode = key[0]
    bucket = cache.get(mode)
    if not bucket:
        return None
    if key in bucket:
        # Move to end (LRU update). dict preserves insertion order in py3.7+.
        val = bucket.pop(key)
        bucket[key] = val
        return val
    return None


def _cache_put(cache, key, value):
    """Store result. Evicts oldest entry in the same mode if over budget."""
    if cache is None:
        return
    mode = key[0]
    bucket = cache.setdefault(mode, {})
    if key in bucket:
        bucket.pop(key)
    bucket[key] = value
    while len(bucket) > CACHE_PER_MODE:
        # Pop oldest (first inserted, since dict preserves order)
        oldest_key = next(iter(bucket))
        bucket.pop(oldest_key)


def _search_cached(cache, qtext, mode, allowed_venues, retrieve_k, rerank_k):
    """search() wrapper that checks/updates the per-session cache."""
    key = _cache_key(mode, qtext, allowed_venues, retrieve_k, rerank_k)
    hit = _cache_get(cache, key)
    if hit is not None:
        print(f"[cache hit] mode={mode}")
        return hit
    print(f"[cache miss] mode={mode}")
    res = search(qtext, mode=mode, allowed_venues=allowed_venues,
                 retrieve_k=retrieve_k, rerank_k=rerank_k)
    _cache_put(cache, key, res)
    return res


# ──────────────────────── Search dispatcher ────────────────────────

def _empty_outputs():
    """Hide all tab slots and clear status."""
    out = []
    for _ in range(MAX_TABS):
        out.append(gr.update(visible=False, label=""))
        out.append("")
        out.append(_EMPTY_GRAPH_HTML)
    out.append("")
    return out


def do_search(cache, venues, retrieve_k, rerank_k,
              bg_q, gap_q, sol_q, method_q, find_q):
    if not _warmup_done.is_set():
        msg = (
            "<div style='color:#9a3412;background:#fff7ed;"
            "border:1px solid #fed7aa;padding:14px 16px;border-radius:8px;'>"
            "&#9203; <b>Server is still warming up.</b> "
            "Please wait ~30 seconds after first launch and try again."
            "</div>"
        )
        out = _empty_outputs()
        out[-1] = msg
        return out + [cache]

    inputs = [
        ("background", "Background", bg_q),
        ("gap", "Gap", gap_q),
        ("solution", "Solution", sol_q),
        ("method", "Approach/Method", method_q),
        ("findings", "Findings", find_q),
    ]
    active = [(m, lbl, q.strip()) for (m, lbl, q) in inputs if q and q.strip()]

    if not active:
        out = _empty_outputs()
        out[-1] = (
            "<div style='color:#6b7280;padding:12px;'>"
            "Fill in at least one component box, then click Search."
            "</div>"
        )
        return out + [cache]

    # Build the search list. Each filled component runs its own mode-specific
    # search. If 2+ are filled, the Combined tab is added below using RRF over
    # the per-mode rankings (no extra search needed).
    searches = []
    for (m, lbl, q) in active:
        searches.append((lbl, m, q))

    selected_venues = list(venues) if venues else None
    rk = max(1, int(rerank_k))
    rt = max(rk, int(retrieve_k))

    out = _empty_outputs()
    error_msgs = []

    # Run all per-mode searches in parallel. vLLM (embedder + reranker) handles
    # concurrent calls via continuous batching.
    def _one_search(item):
        label, mode, qtext = item
        try:
            res = _search_cached(cache, qtext, mode, selected_venues, rt, rk)
            return ("ok", label, mode, qtext, res, None)
        except Exception as e:
            return ("err", label, mode, qtext, None, e)

    todo = list(searches[:MAX_TABS])
    results_by_idx = {}
    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            future_to_idx = {
                ex.submit(_one_search, item): idx
                for idx, item in enumerate(todo)
            }
            for fut, idx in future_to_idx.items():
                results_by_idx[idx] = fut.result()

        for i in range(len(todo)):
            status, label, mode, qtext, res, err = results_by_idx[i]
            if status == "ok":
                results = res["results"]
                list_html = _render_list(results, search_context={
                    "mode": mode,
                    "mode_label": label,
                    "query": qtext,
                })
                graph_html = _build_graph_html(
                    qtext, res["query_embedding"], results
                )
                tab_label = label + " (" + str(len(results)) + ")"
                out[i * 3 + 0] = gr.update(visible=True, label=tab_label)
                out[i * 3 + 1] = list_html
                out[i * 3 + 2] = graph_html
            else:
                error_msgs.append(label + ": " + str(err))
                out[i * 3 + 0] = gr.update(visible=True, label=label + " (error)")
                out[i * 3 + 1] = (
                    "<div style='color:#b91c1c;padding:12px;'>"
                    + html.escape(str(err)) + "</div>"
                )
                out[i * 3 + 2] = _EMPTY_GRAPH_HTML

    # ── Combined (RRF) — only when 2+ component searches succeeded ──
    if todo:
        successful = []  # list of (label, results)
        for i in range(len(todo)):
            status, label, mode, qtext, res, err = results_by_idx[i]
            if status == "ok":
                successful.append((label, res["results"]))

        if len(successful) >= 2:
            fused = _combine_rrf(successful, rk)
            # Move the per-mode tabs one slot to the right and put Combined at slot 0.
            kept = []
            for i in range(len(todo)):
                kept.append((out[i * 3 + 0], out[i * 3 + 1], out[i * 3 + 2]))
            for i in range(MAX_TABS):
                out[i * 3 + 0] = gr.update(visible=False, label="")
                out[i * 3 + 1] = ""
                out[i * 3 + 2] = _EMPTY_GRAPH_HTML
            out[0] = gr.update(visible=True, label="Combined (" + str(len(fused)) + ")")
            out[1] = _render_list_combined(fused)
            out[2] = (
                "<div style='color:#6b7280;padding:32px;text-align:center;'>"
                "Graph view is not available for the Combined tab "
                "(it merges rankings from multiple modes)."
                "</div>"
            )
            for i, (tab_upd, list_html, graph_html) in enumerate(kept, start=1):
                if i >= MAX_TABS:
                    break
                out[i * 3 + 0] = tab_upd
                out[i * 3 + 1] = list_html
                out[i * 3 + 2] = graph_html

    if error_msgs:
        out[-1] = (
            "<div style='color:#b91c1c;padding:8px;'>"
            + " | ".join(html.escape(m) for m in error_msgs)
            + "</div>"
        )
    return out + [cache]


# ──────────────────────── UI layout ────────────────────────

with gr.Blocks(title="HCI Paper Semantic Search", css='.hcips-hidden { display: none !important; visibility: hidden !important; height: 0 !important; width: 0 !important; margin: 0 !important; padding: 0 !important; border: 0 !important; overflow: hidden !important; }', js='() => {\n  console.log(\'hcips: js= function invoked\');\n\n  if (!window.__hcipsListenerAttached) {\n    window.__hcipsListenerAttached = true;\n    window.__hcipsCollectedDois = window.__hcipsCollectedDois || [];\n\n    window.addEventListener(\'message\', function(ev) {\n      var d = ev.data;\n      if (!d || !d.__hcips) return;\n      if (d.type !== \'toggleStar\' && d.type !== \'saveNote\') return;\n      console.log(\'hcips: \' + d.type + \' accepted\', d.doi);\n\n      var boxId, btnId;\n      if (d.type === \'saveNote\') {\n        boxId = \'hcips_note_event\';\n        btnId = \'hcips_note_submit\';\n      } else {\n        boxId = \'hcips_star_event\';\n        btnId = \'hcips_star_submit\';\n        var dois = window.__hcipsCollectedDois;\n        var idx = dois.indexOf(d.doi);\n        if (d.active && idx < 0) dois.push(d.doi);\n        if (!d.active && idx >= 0) dois.splice(idx, 1);\n      }\n\n      function findEventBox(id) {\n        var wrap = document.getElementById(id);\n        if (wrap) {\n          var ta = wrap.querySelector(\'textarea, input\');\n          if (ta) return ta;\n        }\n        return document.querySelector(\'[id*="\' + id + \'"] textarea, [id*="\' + id + \'"] input\');\n      }\n\n      var ta = findEventBox(boxId);\n      if (!ta) { console.warn(\'hcips: event box not found:\', boxId); return; }\n      var proto = (ta.tagName === \'TEXTAREA\') ? window.HTMLTextAreaElement.prototype\n                                              : window.HTMLInputElement.prototype;\n      var nativeSetter = Object.getOwnPropertyDescriptor(proto, \'value\').set;\n      nativeSetter.call(ta, JSON.stringify(d));\n      ta.dispatchEvent(new Event(\'input\', { bubbles: true }));\n      setTimeout(function() {\n        var btn = document.getElementById(btnId);\n        if (btn) { btn.click(); console.log(\'hcips: \' + btnId + \' clicked\'); }\n        else { console.warn(\'hcips: button not found:\', btnId); }\n      }, 50);\n    });\n    console.log(\'hcips: message listener attached\');\n  }\n\n  window.__hcipsParentReady = true;\n}') as demo:
    gr.Markdown("# HCI Paper Semantic Search")

    # Per-session cache store (dict[mode -> dict[key -> result]])
    search_cache_state = gr.State(value={})
    # Per-session collection of starred papers
    collection_state = gr.State(value=[])

    # Hidden textbox that receives JSON payloads from iframe star clicks
    star_event_box = gr.Textbox(elem_id="hcips_star_event", elem_classes=["hcips-hidden"])
    star_event_submit = gr.Button("submit_star", elem_id="hcips_star_submit", elem_classes=["hcips-hidden"])
    note_event_box = gr.Textbox(elem_id="hcips_note_event", elem_classes=["hcips-hidden"])
    note_event_submit = gr.Button("submit_note", elem_id="hcips_note_submit", elem_classes=["hcips-hidden"])

    warmup_banner = gr.HTML(visible=True)

    def _warmup_status():
        if _warmup_done.is_set():
            return gr.update(value="", visible=False)
        return gr.update(
            value=(
                "<div style='color:#9a3412;background:#fff7ed;"
                "border:1px solid #fed7aa;padding:10px 14px;"
                "border-radius:8px;font-size:0.92em;'>"
                "&#9203; <b>Warming up the search engine</b> "
                "&mdash; first launch takes ~30s. "
                "Search will be available shortly."
                "</div>"
            ),
            visible=True,
        )

    timer = gr.Timer(2.0)
    timer.tick(fn=_warmup_status, inputs=None, outputs=warmup_banner)
    demo.load(fn=_warmup_status, inputs=None, outputs=warmup_banner)

    gr.Markdown(
        "_Fill in any of the boxes below for the aspects you want to match. "
        "Boxes you leave empty are skipped. If you fill in two or more, "
        "a **Combined** tab will also appear, ranking papers that "
        "match across all of your filled aspects._"
    )

    component_inputs = []
    for label, key in COMPONENT_BOXES:
        tb = gr.Textbox(label=label, lines=1)
        component_inputs.append(tb)

    venue_filter = gr.CheckboxGroup(
        choices=_VENUES, value=_VENUES, label="Venue",
    )

    with gr.Row():
        retrieve_k_slider = gr.Slider(
            minimum=100, maximum=5000, value=1000, step=100,
            label="Retrieval candidates (embedding stage)",
        )
        rerank_k_slider = gr.Slider(
            minimum=10, maximum=500, value=100, step=10,
            label="Final results (rerank stage)",
        )

    btn = gr.Button("Search", variant="primary")

    with gr.Accordion("My Collection (empty)", open=False) as collection_accordion:
        collection_html = gr.HTML(value=_render_collection([]))
        export_btn = gr.Button("Export .bib", size="sm")
        export_file = gr.File(label="Download", visible=False, interactive=False)

    status = gr.HTML()

    # Pre-allocate MAX_TABS slots; do_search() toggles visibility/labels.
    tab_handles = []
    list_outputs = []
    graph_outputs = []

    with gr.Tabs():
        for i in range(MAX_TABS):
            with gr.Tab(label="_slot_" + str(i), visible=False) as t:
                with gr.Tabs():
                    with gr.Tab("List"):
                        lh = gr.HTML()
                    with gr.Tab("Graph"):
                        gh = gr.HTML(value=_EMPTY_GRAPH_HTML)
            tab_handles.append(t)
            list_outputs.append(lh)
            graph_outputs.append(gh)

    outputs = []
    for i in range(MAX_TABS):
        outputs.append(tab_handles[i])
        outputs.append(list_outputs[i])
        outputs.append(graph_outputs[i])
    outputs.append(status)
    outputs.append(search_cache_state)

    inputs = [search_cache_state, venue_filter, retrieve_k_slider, rerank_k_slider] + component_inputs

    btn.click(fn=do_search, inputs=inputs, outputs=outputs)
    for tb in component_inputs:
        tb.submit(fn=do_search, inputs=inputs, outputs=outputs)


    def _on_collection_change(collection):
        # Re-render the collection panel + accordion label.
        return (
            _render_collection(collection),
            gr.update(label=_collection_label(collection)),
        )

    star_event_submit.click(
        fn=_toggle_collection,
        inputs=[collection_state, star_event_box],
        outputs=[collection_state],
    ).then(
        fn=_on_collection_change,
        inputs=[collection_state],
        outputs=[collection_html, collection_accordion],
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
          var dois = window.__hcipsCollectedDois || [];
          var iframes = document.querySelectorAll('iframe');
          iframes.forEach(function(f) {
            try { f.contentWindow.postMessage({__hcipsSync: true, dois: dois}, '*'); }
            catch (e) {}
          });
          return [];
        }"""
    )

    # Note-save handler: updates user_note silently. We do NOT re-render the
    # collection panel here because that would clobber any other in-progress
    # textareas. The note value is already in the DOM where the user typed it.
    note_event_submit.click(
        fn=_save_note,
        inputs=[collection_state, note_event_box],
        outputs=[collection_state],
    )

    def _do_export(collection):
        path = write_bib_to_tempfile(collection or [])
        return gr.update(value=path, visible=True)

    export_btn.click(
        fn=_do_export,
        inputs=[collection_state],
        outputs=[export_file],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)