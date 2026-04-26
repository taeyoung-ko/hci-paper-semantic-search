"""인덱스 빌드 (one-shot). 컨테이너 내부에서 한 번 실행."""
import json
import os
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from api_clients import embed_documents

DATA_PATH = Path(os.environ["DATA_PATH"])
INDEX_DIR = Path(os.environ["INDEX_DIR"])
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 인덱스에 포함할 venue들. JSONL의 'venue' 필드와 정확히 일치해야 함.
ALLOWED_VENUES = {"CHI", "UIST", "CSCW", "DIS", "IUI", "RecSys", "UMAP"}


def build_text(p: dict) -> str:
    """임베딩 본문: title + abstract."""
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def main() -> None:
    papers, texts = [], []
    venue_counts: dict[str, int] = {}

    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            venue = p.get("venue", "")
            if venue not in ALLOWED_VENUES:
                continue
            text = build_text(p)
            if not text:
                continue
            papers.append(p)
            texts.append(text)
            venue_counts[venue] = venue_counts.get(venue, 0) + 1

    print(f"[build_index] {len(texts)} papers selected from venues:")
    for venue, count in sorted(venue_counts.items(), key=lambda x: -x[1]):
        print(f"  {venue:<10} {count:>6}")

    if not texts:
        raise SystemExit(
            "No papers matched the allowed venues. "
            "Check ALLOWED_VENUES against actual venue values in your JSONL."
        )

    print(f"[build_index] encoding {len(texts)} papers...")
    embs_list: list[list[float]] = []
    BATCH = 64
    for i in tqdm(range(0, len(texts), BATCH)):
        embs_list.extend(embed_documents(texts[i:i + BATCH], batch_size=BATCH))

    embs = np.asarray(embs_list, dtype="float32")
    faiss.normalize_L2(embs)
    print(f"[build_index] dim={embs.shape[1]}, vectors={embs.shape[0]}")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(INDEX_DIR / "sigchi.faiss"))

    with (INDEX_DIR / "sigchi_meta.jsonl").open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[build_index] saved → {INDEX_DIR}")


if __name__ == "__main__":
    main()