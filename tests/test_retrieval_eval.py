# -*- coding: utf-8 -*-
import difflib
from pathlib import Path

from lightrag.product_eval import _metrics, evaluate, load_dataset, recommend_floors
from lightrag.retrieval_admission import admission_probability
from lightrag.search_strategies import tokenize


def test_admission_probability_is_not_the_cosine():
    score = admission_probability(0.45, 0.25, 0.08)
    assert score is not None
    assert abs(score - 0.45) > 0.1
    assert score > 0.5


def test_hybrid_ratios_are_fit_separately():
    def grouped(low, high):
        return {"positive": [high], "hard_negative": [low], "irrelevant": [low]}

    rules = recommend_floors({
        "vector": grouped(0.1, 0.8),
        "keyword": grouped(0.0, 0.3),
        "mix": grouped(0.1, 0.8),
        "tree": grouped(0.1, 0.8),
        "graph": grouped(0.0, 1.0),
        "hybrid": {
            "0": grouped(0.0, 0.3),
            "0.25": grouped(0.02, 0.4),
            "0.5": grouped(0.05, 0.55),
            "0.75": grouped(0.08, 0.7),
            "1": grouped(0.1, 0.8),
        },
    })
    assert rules["hybrid"]["floors"]["0"] < rules["vector"]["floor"]
    assert rules["hybrid"]["floors"]["0"] < rules["hybrid"]["floors"]["1"]
    assert rules["hybrid"]["signal"] == "blend"
    assert "0.15" not in Path(__file__).resolve().parents[1].joinpath("lightrag/retrieval_admission.py").read_text(encoding="utf-8")


def test_missing_citation_and_any_false_return_are_counted():
    missing = _metrics([{
        "answerable": True,
        "expected_units": ["t0001"],
        "expected_docs": ["doc-a"],
        "kb_id": "kb-main",
        "owner_id": "user-a",
        "chunks": [],
        "latency_ms": 1,
    }])
    assert missing["citation_accuracy"] == 0
    leaked = _metrics([{
        "answerable": False,
        "expected_units": [],
        "expected_docs": [],
        "kb_id": "kb-main",
        "owner_id": "user-a",
        "chunks": [{"chunk_id": "graph:x", "kb_id": "kb-other", "owner_id": "user-b", "document_id": "note"}],
        "latency_ms": 1,
    }])
    assert leaked["false_return_rate"] == 1
    assert leaked["cross_kb_leaks"] == 1


def test_holdout_dataset_contract():
    dataset = load_dataset(Path(__file__).parent / "fixtures" / "retrieval_eval.json")
    documents = dataset["documents"]
    questions = dataset["questions"]
    assert len(documents) >= 20
    assert len(questions) >= 100
    assert len([item for item in questions if not item["answerable"]]) >= 30
    assert {item["split"] for item in questions} == {"calibration", "holdout"}
    tracks = {item["track"] for item in questions}
    assert {"semantic", "hard_negative", "near_id", "same_name", "unanswerable", "lexical", "multihop"} <= tracks
    by_id = {item["id"]: item for item in documents}
    for item in questions:
        if item["track"] != "semantic":
            continue
        doc_tokens = set()
        for doc_id in item["expected_docs"]:
            for chunk in by_id[doc_id]["chunks"]:
                doc_tokens.update(tokenize(chunk["text"]))
        assert not (set(tokenize(item["query"])) & doc_tokens)
    calibration = [item["query"] for item in questions if item["split"] == "calibration"]
    holdout = [item["query"] for item in questions if item["split"] == "holdout"]
    for left in calibration:
        for right in holdout:
            assert difflib.SequenceMatcher(None, left, right).ratio() < 0.8


def test_ci_lexical_holdout_targets():
    dataset = load_dataset(Path(__file__).parent / "fixtures" / "retrieval_eval.json")
    report = evaluate(dataset, profile="ci", fit_calibration=False, tracks={"lexical", "unanswerable", "multihop"})
    for mode in ("vector", "keyword", "hybrid", "mix", "tree"):
        metrics = report["holdout"][mode]
        assert metrics["cross_kb_leaks"] == 0, (mode, metrics)
        assert metrics["false_return_rate"] == 0, (mode, metrics)
        assert metrics["recall_at_5"] >= 0.9, (mode, metrics)
        assert metrics["no_answer_f1"] >= 0.9, (mode, metrics)
        assert metrics["citation_accuracy"] == 1, (mode, metrics)
    for key, metrics in report["hybrid_ratios"].items():
        assert metrics["false_return_rate"] == 0, (key, metrics)
        assert metrics["recall_at_5"] >= 0.9, (key, metrics)
    graph = evaluate(dataset, profile="ci", fit_calibration=False, tracks={"multihop", "unanswerable"})
    assert graph["holdout"]["graph"]["cross_kb_leaks"] == 0
    assert graph["holdout"]["graph"]["false_return_rate"] == 0
    assert graph["holdout"]["graph"]["multihop_hit_rate"] == 1
    assert graph["holdout"]["graph"]["citation_accuracy"] == 1
