"""vLLM OpenAI-compatible 엔드포인트 호출 래퍼."""
import os
import requests
from openai import OpenAI

EMBEDDER_URL = os.environ["EMBEDDER_URL"]
RERANKER_URL = os.environ["RERANKER_URL"]

_embed_client = OpenAI(base_url=EMBEDDER_URL, api_key="EMPTY")


# ──────────────────────────────────────────────────────────────────────
# Mode-aware instructions
#
# Each search mode pairs a (1) DOCUMENT-side embedding instruction used at
# index-build time, (2) QUERY-side embedding instruction used at search
# time, and (3) RERANKER instruction used during cross-encoder rescoring.
#
# Qwen3-Embedding is instruction-aware: passing different instructions
# while encoding the same text shifts which aspects of the text get
# weight. Reranker is also instruction-aware, so we keep both stages in
# sync per mode.
# ──────────────────────────────────────────────────────────────────────

MODES = ["topic", "background", "gap", "solution", "method", "findings"]
DEFAULT_MODE = "topic"

# Document-side instruction at index-build time. Keep concise; Qwen
# embedding model expects an "instruct: ... | text: ..." style framing
# but the OpenAI-compatible /v1/embeddings endpoint takes plain strings,
# so we prepend the instruction inline to the document text.
EMBED_DOC_INSTRUCTION = {
    "topic": (
        "Represent this academic paper for semantic search by topic and "
        "research domain."
    ),
    "background": (
        "Represent the research background and domain context of this "
        "academic paper."
    ),
    "gap": (
        "Represent the research gap and limitations of prior work that "
        "this academic paper addresses."
    ),
    "solution": (
        "Represent the proposed solution or system that this academic "
        "paper introduces."
    ),
    "method": (
        "Represent the methodology, technique, or approach used in this "
        "academic paper."
    ),
    "findings": (
        "Represent the experimental findings and results reported in "
        "this academic paper."
    ),
}

# Query-side instruction. Phrased as a retrieval task ("Given X, retrieve
# papers whose Y matches"). Qwen3-Embedding's recommended format is
# "Instruct: <task>\nQuery:".
EMBED_QUERY_INSTRUCTION = {
    "topic": (
        "Given a research description, retrieve academic papers that "
        "address the same topic, problem, or research domain."
    ),
    "background": (
        "Given a research description, retrieve academic papers that "
        "share the same research background and domain context."
    ),
    "gap": (
        "Given a research description, retrieve academic papers that "
        "identify the same kind of research gap or limitations of prior "
        "work."
    ),
    "solution": (
        "Given a research description, retrieve academic papers that "
        "propose a similar solution, system, or artifact."
    ),
    "method": (
        "Given a research description, retrieve academic papers that "
        "use a similar methodology, technique, or approach."
    ),
    "findings": (
        "Given a research description, retrieve academic papers that "
        "report similar experimental findings or results."
    ),
}

# Reranker instruction: cross-encoder gets (query, doc) pair and judges
# match according to this instruction.
RERANK_INSTRUCTION = {
    "topic": (
        "Given a research description, judge whether the academic paper "
        "is relevant to the user's research topic, problem, or domain."
    ),
    "background": (
        "Given a research description, judge whether the academic paper "
        "shares the same research background or domain context."
    ),
    "gap": (
        "Given a research description, judge whether the academic paper "
        "addresses a similar research gap or limitation of prior work."
    ),
    "solution": (
        "Given a research description, judge whether the academic paper "
        "proposes a similar solution, system, or artifact."
    ),
    "method": (
        "Given a research description, judge whether the academic paper "
        "uses a similar methodology, technique, or approach, regardless "
        "of domain."
    ),
    "findings": (
        "Given a research description, judge whether the academic paper "
        "reports similar experimental findings or results."
    ),
}


# Reranker prompt template — must remain exactly this format for Qwen3-Reranker.
RERANK_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _instruct_doc_text(text: str, mode: str) -> str:
    """Prepend mode instruction to a document for the document-side encode."""
    instr = EMBED_DOC_INSTRUCTION[mode]
    return f"Instruct: {instr}\n{text}"


def embed_documents(texts: list[str], mode: str = DEFAULT_MODE,
                    batch_size: int = 64) -> list[list[float]]:
    """Mode-aware document-side embedding."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        instructed = [_instruct_doc_text(t, mode) for t in chunk]
        resp = _embed_client.embeddings.create(
            model="qwen3-embed",
            input=instructed,
        )
        out.extend(d.embedding for d in resp.data)
    return out


def embed_query(query: str, mode: str = DEFAULT_MODE) -> list[float]:
    """Mode-aware query-side embedding."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    instr = EMBED_QUERY_INSTRUCTION[mode]
    formatted = f"Instruct: {instr}\nQuery:{query}"
    resp = _embed_client.embeddings.create(
        model="qwen3-embed",
        input=[formatted],
    )
    return resp.data[0].embedding


def rerank(query: str, docs: list[str], mode: str = DEFAULT_MODE,
           batch_size: int = 32) -> list[float]:
    """Mode-aware cross-encoder rescoring. Returns list of 0..1 scores
    in the same order as docs."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    instr = RERANK_INSTRUCTION[mode]
    scores: list[float] = []
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        text_1 = [
            f"{RERANK_PREFIX}<Instruct>: {instr}\n<Query>: {query}\n"
            for _ in chunk
        ]
        text_2 = [f"<Document>: {d}{RERANK_SUFFIX}" for d in chunk]
        resp = requests.post(
            f"{RERANKER_URL}/score",
            json={"model": "qwen3-rerank", "text_1": text_1, "text_2": text_2},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        scores.extend(float(x["score"]) for x in data)
    return scores
