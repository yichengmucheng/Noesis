from lightrag.open_ie import personalized_pagerank, synonym_targets


def test_synonym_targets_skip_self_and_weak_hits():
    picked = synonym_targets(
        "文成",
        [("文成", 0.99), ("文程", 0.86), ("天津", 0.4), ("文成", 0.2)],
        threshold=0.8,
    )
    assert picked == [("文程", 0.86)]


def test_synonym_targets_convert_distance_metric():
    picked = synonym_targets(
        "文成",
        [("文成", 0.05), ("文程", 0.12)],
        threshold=0.8,
    )
    assert picked == [("文程", 0.88)]


def test_pagerank_spreads_weight_to_neighbors():
    neighbors = {"甲": ["乙"], "乙": ["甲", "丙"], "丙": ["乙"], "丁": []}
    scores = personalized_pagerank(neighbors, {"甲": 1.0}, alpha=0.85, steps=30)
    assert scores["乙"] > scores["丁"]
    assert scores["丙"] > scores["丁"]
    assert scores["甲"] > scores["丁"]
    assert abs(sum(scores.values()) - 1) < 0.05
