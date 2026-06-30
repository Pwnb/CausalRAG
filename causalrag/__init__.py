"""CausalRAG: integrating causal graphs into retrieval-augmented generation.

Reference implementation of Wang et al., "CausalRAG: Integrating Causal Graphs
into Retrieval-Augmented Generation", Findings of ACL 2025.

Public API
----------
build_graph(source, out_path, ...)    Build the text graph from raw text (Sec. 4.1).
query_causal_rag(graph_path, query)   Answer a question over the graph (Sec. 4.2-4.3).
run_single(example, graph_path, ...)  Same, accepting {"question": ...}.
load_graph(graph_path)                Load a built graph as (nodes, relationships).
"""

from .graph import Node, Relationship, build_graph, load_graph
from .method import query_causal_rag, run_single

__all__ = [
    "build_graph",
    "query_causal_rag",
    "run_single",
    "load_graph",
    "Node",
    "Relationship",
]
__version__ = "1.0.0"
