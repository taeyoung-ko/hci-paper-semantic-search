"""vLLM OpenAI-compatible 엔드포인트 호출 래퍼."""
import os
import requests
from openai import OpenAI

EMBEDDER_URL = os.environ["EMBEDDER_URL"]
RERANKER_URL = os.environ["RERANKER_URL"]

# Embedder는 표준 OpenAI embeddings API 사용
_embed_client = OpenAI(base_url=EMBEDDER_URL, api_key="EMPTY")

# 임베딩 task instruction (Qwen3-Embedding은 쿼리 측에만 instruction 사용)
EMBED_INSTRUCTION = (
    "Given a research description or topic, retrieve relevant academic "
    "papers from HCI conferences whose title and abstract address the same "
    "research problem, methodology, or application domain."
)

# Reranker 입력 템플릿 (Qwen3-Reranker 공식 포맷, 절대 변경 금지)
RERANK_INSTRUCTION = (
    "Given a research description or topic, judge whether the academic "
    "paper (title + abstract) is relevant to the user's research interest, "
    "either by addressing the same problem/domain or by providing useful "
    "methodology, evaluation design, or related findings."
)
RERANK_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def embed_documents(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """문서 측 임베딩 — instruction 없이 raw 텍스트만."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = _embed_client.embeddings.create(
            model="qwen3-embed",
            input=chunk,
        )
        out.extend(d.embedding for d in resp.data)
    return out


def embed_query(query: str) -> list[float]:
    """쿼리 측 임베딩 — instruction prepend."""
    formatted = f"Instruct: {EMBED_INSTRUCTION}\nQuery:{query}"
    resp = _embed_client.embeddings.create(
        model="qwen3-embed",
        input=[formatted],
    )
    return resp.data[0].embedding


def rerank(query: str, docs: list[str], batch_size: int = 32) -> list[float]:
    """(query, doc) 페어 채점. 입력 순서대로 0~1 score 리스트 반환."""
    scores: list[float] = []
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        text_1 = [
            f"{RERANK_PREFIX}<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n"
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
        # 응답 순서 보장 안 되니 index로 정렬
        data.sort(key=lambda x: x["index"])
        scores.extend(float(x["score"]) for x in data)
    return scores