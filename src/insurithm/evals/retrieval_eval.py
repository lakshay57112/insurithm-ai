"""Retrieval evaluation: Recall@k and MRR against the gold question set.

A retrieved chunk counts as a hit only if it is on the expected page AND
contains the expected passage. Only answerable questions are scored here;
unanswerable ones are for the later unsupported-claim evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..console import ensure_utf8_output
from ..config import DEFAULT_GOLD_FILE, DEFAULT_POLICY_PDF, REPORT_DIR
from ..ingest import ingest_pdf
from ..models import Chunk
from ..retrieval import BM25Retriever, Retriever


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_hit(chunk: Chunk, gold: dict) -> bool:
    return chunk.page == gold["expected_page"] and normalize(gold["expected_passage"]) in normalize(chunk.text)


def load_gold(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)["questions"]


def evaluate(retriever: Retriever, gold: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    answerable = [q for q in gold if q["answerable"]]
    if not answerable:
        raise ValueError("gold set has no answerable questions")
    max_k = max(ks)

    results = []
    for q in answerable:
        hits = retriever.search(q["question"], k=max_k)
        first_rank = next((h.rank for h in hits if is_hit(h.chunk, q)), None)
        results.append(
            {
                "id": q["id"],
                "first_hit_rank": first_rank,
                "expected_page": q["expected_page"],
                "expected_clause": q.get("expected_clause"),
                "top_retrieved": [
                    {"page": h.chunk.page, "clause": h.chunk.clause_id} for h in hits[:3]
                ],
            }
        )

    n = len(results)
    recall = {
        f"recall@{k}": round(sum(1 for r in results if r["first_hit_rank"] and r["first_hit_rank"] <= k) / n, 4)
        for k in ks
    }
    mrr = round(sum(1 / r["first_hit_rank"] for r in results if r["first_hit_rank"]) / n, 4)
    return {
        "retriever": getattr(retriever, "name", type(retriever).__name__),
        "questions_scored": n,
        **recall,
        "mrr": mrr,
        "misses_at_max_k": [r["id"] for r in results if r["first_hit_rank"] is None],
        "per_question": results,
    }


def main() -> None:
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the gold question set.")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PDF))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_FILE))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--out", default=str(REPORT_DIR / "retrieval_bm25.json"))
    args = parser.parse_args()

    chunks = ingest_pdf(args.policy)
    report = evaluate(BM25Retriever(chunks), load_gold(args.gold), tuple(args.ks))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"retriever: {report['retriever']}  questions: {report['questions_scored']}")
    for k in args.ks:
        print(f"  Recall@{k}: {report[f'recall@{k}']:.3f}")
    print(f"  MRR: {report['mrr']:.3f}")
    if report["misses_at_max_k"]:
        print(f"  missed within top {max(args.ks)}: {', '.join(report['misses_at_max_k'])}")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
