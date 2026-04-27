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


def _render_list(results):
    if not results:
        return "<div style='color:#6b7280;padding:12px;'>No results.</div>"
    return "\n".join(_render_card(i + 1, p) for i, p in enumerate(results))


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


def do_search(venues, retrieve_k, rerank_k,
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
        return out

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
        return out

    # Build the search list. If 2+ components were filled, prepend an Overall
    # search whose query is the concatenation of the active components.
    searches = []
    if len(active) >= 2:
        parts = [lbl + ": " + q for (_m, lbl, q) in active]
        overall_query = "\n\n".join(parts)
        searches.append(("Overall", "topic", overall_query))
    for (m, lbl, q) in active:
        searches.append((lbl, m, q))

    selected_venues = list(venues) if venues else None
    rk = max(1, int(rerank_k))
    rt = max(rk, int(retrieve_k))

    out = _empty_outputs()
    error_msgs = []

    for i, (label, mode, qtext) in enumerate(searches[:MAX_TABS]):
        try:
            res = search(qtext, mode=mode, allowed_venues=selected_venues,
                         retrieve_k=rt, rerank_k=rk)
            results = res["results"]
            list_html = _render_list(results)
            graph_html = _build_graph_html(
                qtext, res["query_embedding"], results
            )
            tab_label = label + " (" + str(len(results)) + ")"
            out[i * 3 + 0] = gr.update(visible=True, label=tab_label)
            out[i * 3 + 1] = list_html
            out[i * 3 + 2] = graph_html
        except Exception as e:
            error_msgs.append(label + ": " + str(e))
            out[i * 3 + 0] = gr.update(visible=True, label=label + " (error)")
            out[i * 3 + 1] = (
                "<div style='color:#b91c1c;padding:12px;'>"
                + html.escape(str(e)) + "</div>"
            )
            out[i * 3 + 2] = _EMPTY_GRAPH_HTML

    if error_msgs:
        out[-1] = (
            "<div style='color:#b91c1c;padding:8px;'>"
            + " | ".join(html.escape(m) for m in error_msgs)
            + "</div>"
        )
    return out


# ──────────────────────── UI layout ────────────────────────

with gr.Blocks(title="HCI Paper Semantic Search") as demo:
    gr.Markdown("# HCI Paper Semantic Search")

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
        "an additional **Overall** result combining them will also be shown._"
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

    inputs = [venue_filter, retrieve_k_slider, rerank_k_slider] + component_inputs

    btn.click(fn=do_search, inputs=inputs, outputs=outputs)
    for tb in component_inputs:
        tb.submit(fn=do_search, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)