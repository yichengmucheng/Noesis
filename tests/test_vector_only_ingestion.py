from __future__ import annotations

import inspect

from lightrag import product_ingest


def test_worker_honors_single_chunk_strategy_and_vector_only_kb() -> None:
    source = inspect.getsource(product_ingest.run_ingestion)

    assert 'strategy == "one"' in source
    assert 'get("graph_enabled", True)' in source
    assert "_write_graph" in source
