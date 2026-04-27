"""인덱스 빌드 (one-shot). 6개 모드 인덱스를 모두 빌드한다.

Run inside the app container:
    docker compose run --rm app python build_index.py

Output (in named volume `paper_index`):
    /app/index/sigchi_topic.faiss
    /app/index/sigchi_background.faiss
    /app/index/sigchi_gap.faiss
    /app/index/sigchi_solution.faiss
    /app/index/sigchi_method.faiss
    /app/index/sigchi_findings.faiss
    /app/index/sigchi_meta.jsonl   (shared metadata, one row per paper)
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from api_clients import embed_documents, MODES

DATA_PATH = Path(os.environ["DATA_PATH"])
INDEX_DIR = Path(os.environ["INDEX_DIR"])
INDEX_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VENUES = {"CHI", "UIST", "CSCW", "DIS", "IUI", "RecSys", "UMAP"}


def build_text(p: dict) -> str:
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

    print(f"[build_index] {len(texts)} papers from venues:")
    for venue, count in sorted(venue_counts.items(), key=lambda x: -x[1]):
        print(f"  {venue:<10} {count:>6}")
    if not texts:
        raise SystemExit("No papers matched ALLOWED_VENUES.")

    # Save shared metadata once
    meta_path = INDEX_DIR / "sigchi_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[build_index] saved metadata: {meta_path}")

    # Build one FAISS index per mode
    BATCH = 64
    for mode in MODES:
        print(f"\n[build_index] === mode={mode} ===")
        embs_list: list[list[float]] = []
        for i in tqdm(range(0, len(texts), BATCH), unit="batch"):
            embs_list.extend(
                embed_documents(texts[i:i + BATCH], mode=mode, batch_size=BATCH)
            )
        embs = np.asarray(embs_list, dtype="float32")
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        out_path = INDEX_DIR / f"sigchi_{mode}.faiss"
        faiss.write_index(index, str(out_path))
        print(f"[build_index] saved {out_path} (dim={embs.shape[1]}, "
              f"vectors={embs.shape[0]})")

    print(f"\n[build_index] All {len(MODES)} indexes saved → {INDEX_DIR}")


if __name__ == "__main__":
    main()
