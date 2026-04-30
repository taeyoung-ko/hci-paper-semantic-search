"""
SIGCHI Conference Paper Metadata Collector (Semantic Scholar 버전)
================================================================

Semantic Scholar Bulk Search API를 사용하여 CHI, CSCW, DIS, IUI, RecSys,
UIST, UMAP 7개 SIGCHI 학회 논문의 메타데이터를 수집합니다.

출력 형식은 get_metadata.py (OpenAlex/Crossref 버전)와 호환되는 JSONL:
    venue, conference_name, year, doi, title, authors, abstract,
    keywords, citation_count

`authors`  — 이름 문자열의 리스트 (e.g., ["Alice Lee", "Bob Park"])
`keywords` — Semantic Scholar가 추출한 키워드 문자열 리스트

Load with:  pd.read_json("sigchi_conf_doi.jsonl", lines=True)

Usage:
    pip install requests
    python fetch_uist_papers_lite.py                        # 전체 7개 학회
    python fetch_uist_papers_lite.py --venues UIST CHI      # 특정 학회만
    python fetch_uist_papers_lite.py --years 2022-2026      # 연도 변경
    python fetch_uist_papers_lite.py --resume                # 이어받기
"""

from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Defaults & constants
# ---------------------------------------------------------------------------

S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = (
    "title,authors,abstract,citationCount,year,venue,externalIds,"
    "publicationTypes,publicationDate,url,openAccessPdf,"
    "fieldsOfStudy"
)
DEFAULT_OUTPUT = "sigchi_conf_doi.jsonl"
CHECKPOINT_FILE = ".collect_checkpoint_s2.json"
SLEEP = 1.0  # 페이지 간 대기 (rate limit 방지)

# venue 이름 → (conference_name, Semantic Scholar venue 표기 후보)
# Semantic Scholar의 venue 표기가 정확히 어떤 문자열인지는 실행 후 확인 필요.
# 여러 후보를 시도하거나, 첫 실행 결과의 venue 필드를 확인하여 조정하세요.
VENUE_REGISTRY: dict[str, dict] = {
    "CHI": {
        "conference_name": "ACM Conference on Human Factors in Computing Systems",
        "s2_venue": "CHI",
    },
    "CSCW": {
        "conference_name": "ACM Conference on Computer-Supported Cooperative Work and Social Computing",
        "s2_venue": "CSCW",
    },
    "DIS": {
        "conference_name": "ACM Conference on Designing Interactive Systems",
        "s2_venue": "DIS",
    },
    "IUI": {
        "conference_name": "ACM International Conference on Intelligent User Interfaces",
        "s2_venue": "IUI",
    },
    "RecSys": {
        "conference_name": "ACM Conference on Recommender Systems",
        "s2_venue": "RecSys",
    },
    "UIST": {
        "conference_name": "ACM Symposium on User Interface Software and Technology",
        "s2_venue": "UIST",
    },
    "UMAP": {
        "conference_name": "ACM Conference on User Modeling, Adaptation and Personalization",
        "s2_venue": "UMAP",
    },
}

DEFAULT_VENUES = list(VENUE_REGISTRY.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("s2-collector")


# ---------------------------------------------------------------------------
# Semantic Scholar client
# ---------------------------------------------------------------------------

def fetch_all_papers(s2_venue: str, year_range: str,
                     api_key: str | None = None) -> list[dict]:
    """Bulk Search API로 venue+year에 해당하는 전체 논문을 페이지네이션 수집."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    all_papers: list[dict] = []
    token = None
    page = 1

    while True:
        params: dict = {
            "query": "",
            "venue": s2_venue,
            "year": year_range,
            "fields": S2_FIELDS,
        }
        if token:
            params["token"] = token

        log.info("  페이지 %d 요청 중...", page)

        for attempt in range(4):
            try:
                resp = requests.get(S2_BULK_URL, params=params,
                                    headers=headers, timeout=60)
                if resp.status_code == 200:
                    break
                if resp.status_code == 429:
                    wait = 2 ** attempt * 2
                    log.warning("  Rate limit (429). %ds 대기...", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code in (500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                log.error("  HTTP %d: %s", resp.status_code, resp.text[:200])
                return all_papers
            except requests.RequestException as e:
                log.warning("  네트워크 오류: %s", e)
                time.sleep(2 ** attempt)
        else:
            log.error("  최대 재시도 초과")
            break

        data = resp.json()
        papers = data.get("data", [])
        all_papers.extend(papers)
        log.info("  → %d건 (누적 %d건)", len(papers), len(all_papers))

        token = data.get("token")
        if not token:
            break

        page += 1
        time.sleep(SLEEP)

    return all_papers


# ---------------------------------------------------------------------------
# Row conversion (get_metadata.py 출력 형식에 맞춤)
# ---------------------------------------------------------------------------

def to_row(p: dict, venue: str, conference_name: str) -> dict:
    """Semantic Scholar 응답 1건 → get_metadata.py 호환 JSONL row."""
    authors_raw = p.get("authors") or []
    external_ids = p.get("externalIds") or {}
    doi = (external_ids.get("DOI") or "").lower()

    # keywords: Semantic Scholar의 fieldsOfStudy를 사용
    # (get_metadata.py의 OpenAlex keywords와 유사한 역할)
    keywords = p.get("fieldsOfStudy") or []

    return {
        "venue": venue,
        "conference_name": conference_name,
        "year": p.get("year") or 0,
        "doi": doi,
        "title": (p.get("title") or "").replace("\n", " ").strip(),
        "authors": [a.get("name", "") for a in authors_raw
                    if a.get("name")],
        "abstract": (p.get("abstract") or "").replace("\n", " ").strip(),
        "keywords": keywords,
        # ── get_metadata.py에는 없지만 Semantic Scholar 고유 보너스 필드 ──
        "citation_count": p.get("citationCount") or 0,
    }


def is_likely_paratext(p: dict) -> bool:
    """제목 기반 paratext 휴리스틱 (Semantic Scholar에는 is_paratext 플래그 없음).

    get_metadata.py의 OpenAlex is_paratext 필터에 대응하는 간이 필터.
    """
    title = (p.get("title") or "").lower().strip()
    if not title:
        return True
    paratext_patterns = [
        "proceedings of",
        "table of contents",
        "author index",
        "front matter",
        "back matter",
        "title page",
        "welcome from the",
        "message from the",
        "conference committee",
        "reviewer acknowledgment",
        "reviewers list",
        "editorial board",
        "keynote abstract",
        "session details",
    ]
    return any(title.startswith(pat) or title == pat
               for pat in paratext_patterns)


# ---------------------------------------------------------------------------
# Checkpointing (get_metadata.py와 동일한 패턴)
# ---------------------------------------------------------------------------

def load_checkpoint() -> set[str]:
    p = Path(CHECKPOINT_FILE)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except Exception:
        return set()


def save_checkpoint(done: set[str]) -> None:
    Path(CHECKPOINT_FILE).write_text(json.dumps(sorted(done)))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    from collections import Counter

    log.info("=" * 60)

    # 학회별 통계
    venues = Counter(r["venue"] for r in rows)
    log.info("학회별 논문 수:")
    for venue in sorted(venues.keys()):
        log.info("  %s: %d편", venue, venues[venue])

    # 연도별 통계
    years = Counter(r["year"] for r in rows)
    log.info("연도별 논문 수:")
    for year in sorted(years.keys()):
        subset = [r for r in rows if r["year"] == year]
        avg_cite = sum(r["citation_count"] for r in subset) / len(subset)
        log.info("  %s: %d편 (평균 인용수: %.1f)", year, len(subset), avg_cite)

    log.info("인용수 Top 10:")
    top = sorted(rows, key=lambda r: r["citation_count"], reverse=True)[:10]
    for r in top:
        t = r["title"][:50] + "..." if len(r["title"]) > 50 else r["title"]
        log.info("  [%s %s] %s (%d)", r["venue"], r["year"], t, r["citation_count"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Scholar로 SIGCHI 학회 논문 메타데이터 수집 "
                    "(get_metadata.py 호환 JSONL 출력)")
    parser.add_argument("--venues", nargs="+", default=None,
                        help="수집할 학회 목록. 미지정 시 전체 7개 학회 수집. "
                             f"가능한 값: {', '.join(VENUE_REGISTRY)}")
    parser.add_argument("--years", default="2020-2026",
                        help="연도 범위, e.g. 2020-2026")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=None,
                        help="Semantic Scholar API key (선택)")
    parser.add_argument("--resume", action="store_true",
                        help="체크포인트 기반 이어받기")
    args = parser.parse_args()

    venues = args.venues or DEFAULT_VENUES

    # 유효성 검사
    for v in venues:
        if v not in VENUE_REGISTRY:
            log.error("알 수 없는 venue: %s (가능한 값: %s)",
                      v, ", ".join(VENUE_REGISTRY))
            return

    done = load_checkpoint() if args.resume else set()
    write_mode = "a" if (args.resume and Path(args.output).exists()) else "w"
    all_rows: list[dict] = []

    log.info("대상 학회: %s (%s)", ", ".join(venues), args.years)
    log.info("=" * 60)

    with open(args.output, write_mode, encoding="utf-8") as fout:
        for venue in venues:
            info = VENUE_REGISTRY[venue]
            conference_name = info["conference_name"]
            s2_venue = info["s2_venue"]
            checkpoint_key = f"{venue}|{args.years}"

            if checkpoint_key in done:
                log.info("[%s] 이미 수집 완료 — 건너뜁니다", venue)
                continue

            log.info("[%s] 수집 시작 (s2_venue=%s)", venue, s2_venue)

            raw = fetch_all_papers(s2_venue, args.years, args.api_key)
            if not raw:
                log.warning("[%s] 결과 없음. S2 venue 표기를 확인하세요.", venue)
                continue

            # 변환 & paratext 필터
            rows: list[dict] = []
            skipped_paratext = 0
            for p in raw:
                if is_likely_paratext(p):
                    skipped_paratext += 1
                    continue
                rows.append(to_row(p, venue, conference_name))
            if skipped_paratext:
                log.info("[%s] paratext 필터: %d건 제외", venue, skipped_paratext)

            # JSONL 쓰기
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False))
                fout.write("\n")
            fout.flush()

            done.add(checkpoint_key)
            save_checkpoint(done)

            log.info("[%s] %d건 저장 완료", venue, len(rows))
            all_rows.extend(rows)

    log.info("전체 %d건 수집 완료 → %s", len(all_rows), args.output)
    if all_rows:
        print_summary(all_rows)


if __name__ == "__main__":
    main()