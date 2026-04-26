import html
import gradio as gr
from search import search, get_filter_options

_OPTS = get_filter_options()
_VENUES = _OPTS["venues"]


def _score_to_color(score: float) -> tuple[str, str]:
    s = max(0.0, min(1.0, float(score)))
    hue = int(120 * s)  # 0=red, 120=green
    bg = f"hsl({hue}, 75%, 55%)"
    return bg, "#ffffff"


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
            f'{html.escape(k)}</span>'
            for k in keywords
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


def render_results(query: str, venues: list[str]):
    if not query.strip():
        return "<div style='color:#6b7280;padding:12px;'>Enter a query to search.</div>"

    selected_venues = venues if venues else None

    results = search(
        query,
        allowed_venues=selected_venues,
        retrieve_k=1000,
        rerank_k=100,
    )
    if not results:
        return "<div style='color:#6b7280;padding:12px;'>No results.</div>"
    return "\n".join(_render_card(i + 1, p) for i, p in enumerate(results))


with gr.Blocks(title="HCI Paper Semantic Search") as demo:
    gr.Markdown("# HCI Paper Semantic Search")

    q = gr.Textbox(label="Query", lines=3, placeholder="")

    venue_filter = gr.CheckboxGroup(
        choices=_VENUES,
        value=_VENUES,
        label="Venue",
    )

    btn = gr.Button("Search", variant="primary")

    out = gr.HTML()

    btn.click(
        fn=render_results,
        inputs=[q, venue_filter],
        outputs=out,
    )
    q.submit(
        fn=render_results,
        inputs=[q, venue_filter],
        outputs=out,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
