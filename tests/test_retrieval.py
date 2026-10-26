import pytest

from insurithm.config import DEFAULT_GOLD_FILE, DEFAULT_POLICY_PDF
from insurithm.evals.retrieval_eval import evaluate, load_gold
from insurithm.ingest import ingest_pdf
from insurithm.retrieval import BM25Retriever, stem, tokenize


@pytest.fixture(scope="module")
def retriever():
    return BM25Retriever(ingest_pdf(DEFAULT_POLICY_PDF))


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The claims were rejected") == ["claim", "reject"]
    assert "the" not in tokenize("the claim")
    assert stem("claims") == "claim"
    assert stem("hospitalised") == stem("hospitalisation") == "hospitalis"


def test_search_returns_ranked_hits(retriever):
    hits = retriever.search("waiting period for pre-existing disease", k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_empty_or_stopword_query_returns_nothing(retriever):
    assert retriever.search("", k=3) == []
    assert retriever.search("the and of", k=3) == []


def test_obvious_question_finds_the_right_clause(retriever):
    top = retriever.search("How soon must I tell the insurer after an emergency admission?", k=1)[0]
    assert top.chunk.clause_id == "6.2"
    assert top.chunk.page == 6


def test_baseline_does_not_regress(retriever):
    """Floor set just below the measured baseline (Recall@3 0.933, Recall@5 1.0)."""
    report = evaluate(retriever, load_gold(DEFAULT_GOLD_FILE), ks=(1, 3, 5))
    assert report["recall@5"] >= 0.95
    assert report["recall@3"] >= 0.90
    assert report["mrr"] >= 0.75
