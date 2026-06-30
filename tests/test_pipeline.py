"""Offline end-to-end smoke test for CausalRAG.

Runs the full text -> graph -> QA pipeline with a deterministic mock extractor,
mock LLM, and the hashing embedder fallback (no API key, no network). Run with:

    python tests/test_pipeline.py        # or:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalrag import build_graph, query_causal_rag  # noqa: E402
from causalrag.graph import Node, Relationship  # noqa: E402


def mock_extractor(text: str):
    """Turn a chunk into a tiny chain of entities so the graph is non-empty."""
    words = [w.strip(".,").capitalize() for w in text.split() if len(w) > 4]
    uniq = list(dict.fromkeys(words))[:8]
    nodes = [Node(id=w, type="Concept") for w in uniq]
    rels = [Relationship(source=uniq[i], target=uniq[i + 1], type="influences")
            for i in range(len(uniq) - 1)]
    return nodes, rels


def mock_llm(prompt: str) -> str:
    if "causality analysis report" in prompt or "Network Data" in prompt:
        return "Causal report: aspirin reduces inflammation, which reduces pain."
    return "Aspirin reduces inflammation, which in turn reduces pain."


def test_pipeline() -> None:
    import tempfile

    passages = (
        "Aspirin reduces inflammation in tissues. "
        "Inflammation causes pain and swelling. "
        "Pain signals travel through nerves to the brain."
    )

    with tempfile.TemporaryDirectory() as tmp:
        graph_path = build_graph(
            passages, Path(tmp) / "graph.json", extractor=mock_extractor, chunk_size=80
        )
        result = query_causal_rag(
            str(graph_path),
            "What reduces inflammation?",
            k=3,
            s=2,
            llm=mock_llm,
        )

    assert isinstance(result, dict), type(result)
    assert result["answer"].strip(), result
    assert result["causal_report"].strip(), result
    print("ANSWER         :", result["answer"])
    print("SEED NODES     :", result["seed_nodes"])
    print("RELATIONSHIPS  :", result["meta"]["num_relationships"])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    test_pipeline()
