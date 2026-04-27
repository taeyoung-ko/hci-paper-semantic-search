import html
import numpy as np
import textwrap
import gradio as gr
import plotly.graph_objects as go
import plotly.io as pio
import umap

from search import search, get_filter_options

_OPTS = get_filter_options()
_VENUES = _OPTS["venues"]


# Warm up numba/UMAP at startup so the first user search isn't slow.
# Cost: adds ~30s to container startup, but every search after that is fast.
print("[ui] Warming up UMAP/numba JIT (one-time, ~30s)...")
_dummy = np.random.randn(30, 32).astype("float32")
umap.UMAP(n_components=2, n_neighbors=5, random_state=42).fit_transform(_dummy)
print("[ui] Warmup complete.")


def _score_to_color(score: float) -> tuple[str, str]:
    s = max(0.0, min(1.0, float(score)))
    hue = int(120 * s)
    bg = f"hsl({hue}, 75%, 55%)"
    return bg, "#ffffff"


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _wrap_for_hover(s: str, width: int = 80, max_lines: int = 6) -> str:
    """Word-wrap a string for plotly hover (uses <br>). Caps height with max_lines."""
    s = (s or "").replace("\r", " ").strip()
    if not s:
        return ""
    lines = textwrap.wrap(s, width=width, break_long_words=False,
                          break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "…"
    return "<br>".join(html.escape(line) for line in lines)


# ──────────────────────── List view ────────────────────────

def _render_card(rank: int, p: dict) -> str:
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
        chips = "".join(
            f'<span style="display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;'
            f'background:#eef2f7;color:#374151;border-radius:10px;font-size:0.82em;">'
            f'{html.escape(k)}</span>' for k in keywords
        )
        keywords_html = f'<div style="margin:8px 0 4px 0;">{chips}</div>'

    abstract_html = ""
    if abstract:
        abstract_html = (
            f'<div style="font-size:0.93em;color:#333;line-height:1.55;'
            f'margin-top:6px;white-space:pre-wrap;">{abstract}</div>'
        )

    title_link = (
        f'<a href="https://doi.org/{html.escape(doi)}" target="_blank" '
        f'style="font-weight:600;font-size:1.05em;color:#1d4ed8;'
        f'text-decoration:none;">{title}</a>'
        if doi else
        f'<span style="font-weight:600;font-size:1.05em;color:#111;">{title}</span>'
    )

    score_badge = (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:8px;'
        f'background:{bg};color:{fg};font-weight:700;font-size:0.92em;'
        f'min-width:60px;text-align:center;">{score:.3f}</span>'
    )

    meta = (
        f'<span style="color:#6b7280;">#{rank} · {venue} {year}</span>'
        if (venue or year) else f'<span style="color:#6b7280;">#{rank}</span>'
    )

    return f"""
<div style="border-bottom:1px solid #e5e7eb;padding:16px 6px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
    {score_badge}
    <div style="font-size:0.88em;">{meta}</div>
  </div>
  <div>{title_link}</div>
  <div style="font-size:0.9em;color:#4b5563;margin-top:4px;">{authors_str}</div>
  {keywords_html}
  {abstract_html}
</div>
"""


def _render_list(results: list[dict]) -> str:
    if not results:
        return "<div style='color:#6b7280;padding:12px;'>No results.</div>"
    return "\n".join(_render_card(i + 1, p) for i, p in enumerate(results))


# ──────────────────────── Graph view ────────────────────────

_EMPTY_GRAPH_HTML = (
    "<div style='color:#6b7280;padding:32px;text-align:center;'>"
    "Enter a query and click Search to see the graph.</div>"
)


def _build_graph_html(query: str, query_emb: np.ndarray,
                      results: list[dict]) -> str:
    n = len(results)
    if n == 0 or query_emb is None:
        return ("<div style='color:#6b7280;padding:32px;text-align:center;'>"
                "No results.</div>")

    vecs = np.vstack([query_emb.reshape(-1)] + [p["_vec"] for p in results]).astype("float32")

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
        authors = p.get("authors") or []
        authors_str = html.escape(", ".join(authors)) if authors else "(no authors)"
        venue = html.escape(str(p.get("venue") or ""))
        year = html.escape(str(p.get("year") or ""))
        score = float(p.get("rerank_score", 0.0))
        hover_texts.append(
            f"<b>{title}</b><br>"
            f"{authors_str}<br>"
            f"{venue} {year}<br>"
            f"score {score:.3f}"
        )
        doi = p.get("doi") or ""
        customdata.append(f"https://doi.org/{doi}" if doi else "")

    fig = go.Figure()

    # Edges as one trace (None separators) — much faster than 100 traces
    edge_x: list = []
    edge_y: list = []
    for (x, y) in final_xy:
        edge_x.extend([0.0, float(x), None])
        edge_y.extend([0.0, float(y), None])
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.25)", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=final_xy[:, 0], y=final_xy[:, 1],
        mode="markers",
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
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=24, color="#111", symbol="star",
                    line=dict(color="#fff", width=2)),
        text=["Query"], textposition="top center",
        textfont=dict(size=12, color="#111"),
        hovertext=_truncate(query, 200), hoverinfo="text",
        showlegend=False,
    ))

    # Compute a square data range that fits all points + a small margin.
    # Using explicit ranges (not scaleanchor) makes plotly fill the canvas
    # while still keeping x/y on the same data scale.
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

    div_id = "umap_graph_plot"
    plot_div = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
        default_height="100%", default_width="100%",
    )

    click_js = f"""
<script>
(function() {{
  function attach() {{
    var el = document.getElementById("{div_id}");
    if (!el || !el.on) {{ return setTimeout(attach, 100); }}
    el.on("plotly_click", function(data) {{
      if (!data || !data.points || data.points.length === 0) return;
      var url = data.points[0].customdata;
      if (url) {{ window.open(url, "_blank"); }}
    }});
  }}
  attach();
}})();
</script>
"""
    self_resize_js = """
<script>
(function() {
  function fit() {
    try {
      var fe = window.parent && window.frameElement;
      if (!fe) return;
      var w = fe.getBoundingClientRect().width;
      if (w <= 0) return;
      // Force iframe square
      fe.style.height = w + "px";
      // Tell plotly the new exact pixel size via relayout (most reliable).
      if (window.Plotly) {
        var divs = document.querySelectorAll(".plotly-graph-div");
        divs.forEach(function(d) {
          try {
            window.Plotly.relayout(d, {width: w, height: w, autosize: false});
          } catch (e) {}
        });
      }
    } catch (e) {}
  }
  fit();
  setTimeout(fit, 50);
  setTimeout(fit, 200);
  setTimeout(fit, 600);
  setTimeout(fit, 1500);
  window.addEventListener("resize", fit);
  try {
    var ro = new ResizeObserver(fit);
    if (window.frameElement && window.frameElement.parentElement) {
      ro.observe(window.frameElement.parentElement);
    }
  } catch (e) {}
})();
</script>
"""
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;}}.plotly-graph-div{{width:100%!important;height:100%!important;}}.svg-container{{width:100%!important;height:100%!important;}}</style></head><body style="margin:0;width:100vw;height:100vh;overflow:hidden;">
{plot_div}
{click_js}
{self_resize_js}
</body></html>"""
    # Use srcdoc to embed HTML directly (more permissive than data: URLs).
    # The HTML must be entity-escaped because srcdoc is an HTML attribute.
    import html as _html
    escaped = _html.escape(full_html, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="display:block;width:100%;height:600px;border:0;"></iframe>'
    )


# ──────────────────────── Main ────────────────────────

def do_search(query: str, venues: list[str],
              retrieve_k: float, rerank_k: float):
    if not query.strip():
        return (
            "<div style='color:#6b7280;padding:12px;'>Enter a query to search.</div>",
            _EMPTY_GRAPH_HTML,
        )
    selected_venues = venues if venues else None
    rk = max(1, int(rerank_k))
    rt = max(rk, int(retrieve_k))
    out = search(query, allowed_venues=selected_venues,
                 retrieve_k=rt, rerank_k=rk)
    results = out["results"]
    list_html = _render_list(results)
    graph_html = _build_graph_html(query, out["query_embedding"], results)
    return list_html, graph_html


with gr.Blocks(title="HCI Paper Semantic Search") as demo:
    gr.Markdown("# HCI Paper Semantic Search")

    q = gr.Textbox(label="Query", lines=3, placeholder="")
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

    with gr.Tabs():
        with gr.Tab("List"):
            out_list = gr.HTML()
        with gr.Tab("Graph"):
            gr.Markdown(
                "_Query at center · distance from center = `1 − rerank_score` · "
                "direction preserves UMAP cluster structure · click a node to open DOI._"
            )
            out_graph = gr.HTML(value=_EMPTY_GRAPH_HTML)

    btn.click(
        fn=do_search,
        inputs=[q, venue_filter, retrieve_k_slider, rerank_k_slider],
        outputs=[out_list, out_graph],
    )
    q.submit(
        fn=do_search,
        inputs=[q, venue_filter, retrieve_k_slider, rerank_k_slider],
        outputs=[out_list, out_graph],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
