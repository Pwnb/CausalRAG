"""CausalRAG query-time method (paper Sections 4.2 - 4.3).

Given a built graph and a question:

1. **Causal-path discovery (4.2).** Retrieve the top-``k`` seed nodes by
   embedding similarity and expand ``s`` hops along the graph's edges to form a
   causal subgraph. An LLM then turns that subgraph into a structured causality
   analysis report (``CAUSAL_DISCOVERY_PROMPT``).
2. **Grounded generation (4.3).** A second LLM call merges the report with the
   query into the final, grounded answer (``CAUSAL_SUMMARY_PROMPT``).

Defaults follow the paper: ``gpt-4o-mini``, ``all-MiniLM-L6-v2`` embeddings, and
``k = s = 3``.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from .graph import load_graph
from .llm import DEFAULT_MODEL, LLMCallable, make_openai_llm
from .prompts import CAUSAL_DISCOVERY_PROMPT, CAUSAL_SUMMARY_PROMPT, DEFAULT_RESPONSE_TYPE
from .retrieval import (
    DEFAULT_EMBED_MODEL,
    Embedder,
    expand_subgraph,
    format_network_data,
    make_embedder,
    seed_retrieval,
)


def query_causal_rag(
    graph_path: str,
    query: str,
    *,
    k: int = 3,
    s: int = 3,
    model: str = DEFAULT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    response_type: str = DEFAULT_RESPONSE_TYPE,
    api_key: Optional[str] = None,
    llm: Optional[LLMCallable] = None,
    embedder: Optional[Embedder] = None,
) -> Dict:
    """Answer ``query`` over a CausalRAG graph and return a result dict.

    Returns
    -------
    dict with keys:
        ``answer``          final grounded answer (str)
        ``causal_report``   the intermediate causality analysis report (str)
        ``network_data``    the retrieved subgraph rendered for the prompt (str)
        ``seed_nodes``      ids of the top-k seed nodes
        ``meta``            {num_relationships, k, s, model}
    """
    nodes, relationships = load_graph(graph_path)
    if llm is None:
        llm = make_openai_llm(model=model, api_key=api_key)
    if embedder is None:
        embedder = make_embedder(embed_model)

    # 4.2 (a) top-k seeds + s-hop expansion -> causal subgraph
    seed_ids = seed_retrieval(query, nodes, k, embedder)
    sub_rels = expand_subgraph(relationships, set(seed_ids), s)
    network_data = format_network_data(nodes, seed_ids, sub_rels)

    # 4.2 (b) LLM causal-discovery report over the subgraph
    discovery_prompt = CAUSAL_DISCOVERY_PROMPT.format(graph_data=network_data)
    causal_report = llm(discovery_prompt).strip()

    # 4.3 grounded answer from the causal summary report
    summary_prompt = CAUSAL_SUMMARY_PROMPT.format(
        causal_summary=causal_report,
        response_type=response_type,
        query=query,
    )
    answer = llm(summary_prompt).strip()

    return {
        "answer": answer,
        "causal_report": causal_report,
        "network_data": network_data,
        "seed_nodes": seed_ids,
        "meta": {
            "num_relationships": len(sub_rels),
            "k": k,
            "s": s,
            "model": model,
        },
    }


def run_single(example: Union[Dict, str], graph_path: str, **kwargs) -> Dict:
    """Convenience wrapper: accept ``{"question": ...}`` (or a raw string)."""
    if isinstance(example, dict):
        question = example.get("question") or example.get("gold_question") or ""
    else:
        question = str(example)
    return query_causal_rag(graph_path, question, **kwargs)
