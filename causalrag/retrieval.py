"""Retrieval: embed nodes, pick top-k seeds, expand s hops (paper Section 4.2).

This mirrors the original CausalRAG retriever:

1. Embed every graph node with ``all-MiniLM-L6-v2`` (384-dim) and index them in
   a FAISS index using cosine similarity (paper Appendix C). A NumPy fallback is
   used when FAISS / sentence-transformers are unavailable.
2. Retrieve the top-``k`` initial nodes closest to the query embedding.
3. Expand ``s`` hops along the graph's relationships, collecting the directed
   relationships traversed. These nodes + relationships form the causal
   subgraph that is summarised by the LLM.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .graph import Node, Relationship

# An embedder maps a list of strings to a 2-D float array (n, dim).
Embedder = Callable[[Sequence[str]], np.ndarray]

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _hashing_embed(texts: Sequence[str], dim: int = 384) -> np.ndarray:
    """Deterministic dependency-free embedding (fallback when no ST backend)."""
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for tok in str(text).lower().split():
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            vecs[i, h % dim] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def make_embedder(embed_model: str = DEFAULT_EMBED_MODEL) -> Embedder:
    """Return a SentenceTransformer embedder, or a hashing fallback."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embed_model)

        def _embed(texts: Sequence[str]) -> np.ndarray:
            return np.asarray(model.encode(list(texts)), dtype="float32")

        return _embed
    except Exception:
        return lambda texts: _hashing_embed(texts)


# --------------------------------------------------------------------------- #
# Top-k seed retrieval (FAISS cosine similarity, with NumPy fallback)
# --------------------------------------------------------------------------- #
def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype("float32")


def _topk_indices(query_vec: np.ndarray, node_vecs: np.ndarray, k: int) -> List[int]:
    """Return indices of the ``k`` nodes most similar to the query (cosine).

    Vectors are L2-normalised, so inner product equals cosine similarity
    (paper Appendix C); a FAISS ``IndexFlatIP`` is used when available.
    """
    k = max(1, min(k, len(node_vecs)))
    node_vecs = _l2_normalize(node_vecs)
    query_vec = _l2_normalize(query_vec)
    try:
        import faiss

        index = faiss.IndexFlatIP(node_vecs.shape[1])
        index.add(node_vecs)
        _, idx = index.search(query_vec, k)
        return [int(i) for i in idx[0] if i != -1]
    except Exception:
        sims = node_vecs @ query_vec[0]
        return [int(i) for i in np.argsort(-sims)[:k]]


def seed_retrieval(
    query: str,
    nodes: List[Node],
    k: int,
    embedder: Embedder,
) -> List[str]:
    """Return the ids of the top-``k`` nodes most similar to the query."""
    if not nodes:
        return []
    node_texts = [f"Node: {n.id}, Type: {n.type}, Properties: {n.properties}" for n in nodes]
    node_vecs = embedder(node_texts)
    query_vec = embedder([query])
    indices = _topk_indices(query_vec, node_vecs, k)
    return [nodes[i].id for i in indices]


# --------------------------------------------------------------------------- #
# s-hop subgraph expansion (port of the original CausalRAG retriever)
# --------------------------------------------------------------------------- #
def expand_subgraph(
    relationships: List[Relationship],
    initial_node_ids: Set[str],
    s: int,
) -> List[Dict]:
    """Expand ``s`` hops from the seeds, collecting traversed relationships.

    A relationship is traversed when either endpoint is in the current frontier;
    ``direction`` records whether it was followed forward (from source) or in
    reverse (from target), preserving the original cause -> effect orientation.
    """
    current_hop: Set[str] = set(initial_node_ids)
    visited: Set[str] = set(initial_node_ids)
    collected: List[Dict] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for _ in range(max(0, s)):
        next_hop: Set[str] = set()
        for rel in relationships:
            src_in = rel.source in current_hop
            tgt_in = rel.target in current_hop
            if not (src_in or tgt_in):
                continue
            if src_in:
                source_node, extended_node, direction = rel.source, rel.target, "forward"
            else:
                source_node, extended_node, direction = rel.target, rel.source, "reverse"
            key = (source_node, extended_node, rel.type, direction)
            if key not in seen:
                collected.append(
                    {"source": source_node, "target": extended_node, "type": rel.type, "direction": direction}
                )
                seen.add(key)
            next_hop.add(extended_node)
        current_hop = next_hop - visited
        visited |= next_hop
        if not current_hop:
            break
    return collected


# --------------------------------------------------------------------------- #
# Format the subgraph as "Network Data" for the causal-discovery prompt
# --------------------------------------------------------------------------- #
def format_network_data(
    nodes: List[Node],
    seed_ids: List[str],
    relationships: List[Dict],
) -> str:
    """Render the retrieved entities + causal relationships as plain text."""
    meta = {n.id: n for n in nodes}
    involved: List[str] = []
    seen_ids: Set[str] = set()
    for nid in list(seed_ids) + [rid for rel in relationships for rid in (rel["source"], rel["target"])]:
        if nid in meta and nid not in seen_ids:
            involved.append(nid)
            seen_ids.add(nid)

    lines: List[str] = ["**Entities:**"]
    for nid in involved:
        lines.append(f"- {nid} (type: {meta[nid].type})")

    lines.append("")
    lines.append("**Causal Relationships:**")
    if relationships:
        seen_desc: Set[str] = set()
        for rel in relationships:
            # expand_subgraph stores the *traversed* orientation; restore the
            # graph's true cause -> effect direction (reverse hops are flipped).
            if rel["direction"] == "forward":
                src, tgt = rel["source"], rel["target"]
            else:
                src, tgt = rel["target"], rel["source"]
            src_type = meta.get(src).type if src in meta else "entity"
            tgt_type = meta.get(tgt).type if tgt in meta else "entity"
            desc = f"{src} (a {src_type}) --[{rel['type']}]--> {tgt} (a {tgt_type})"
            if desc not in seen_desc:
                lines.append(f"- {desc}")
                seen_desc.add(desc)
    else:
        lines.append("- (no relationships retrieved)")
    return "\n".join(lines)
