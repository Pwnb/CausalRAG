"""Indexing: text -> entity/relation graph (paper Section 4.1).

Following the paper, the text graph is built with LangChain's
``LLMGraphTransformer`` (Chase, 2022): an LLM parses each text chunk into
entities (nodes) and relationships (edges). The result is normalised into a
small, dependency-free JSON schema that the retriever consumes:

    {
      "nodes":         [{"id", "type", "properties"}],
      "relationships": [{"source", "target", "type"}]
    }

The LLM extractor is pluggable via ``extractor=`` (a callable
``(text) -> (nodes, relationships)``) so the graph can be built offline or with
a different backend in tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .llm import DEFAULT_MODEL, require_api_key


@dataclass
class Node:
    id: str
    type: str = "Entity"
    properties: Dict = field(default_factory=dict)


@dataclass
class Relationship:
    source: str
    target: str
    type: str = "related"


# An extractor maps a chunk of text to (nodes, relationships).
Extractor = Callable[[str], Tuple[List[Node], List[Relationship]]]


# --------------------------------------------------------------------------- #
# Input loading + chunking
# --------------------------------------------------------------------------- #
def _read_text(source) -> str:
    """Accept raw text, a list of strings, or a path to a .txt/.pdf file."""
    if isinstance(source, (list, tuple)):
        return "\n".join(str(s) for s in source)
    path = Path(source) if isinstance(source, (str, Path)) else None
    if path is not None and path.exists():
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return " ".join((page.extract_text() or "") for page in reader.pages)
        return path.read_text(encoding="utf-8", errors="ignore")
    return str(source)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Character-window chunking with overlap (mirrors LangChain's splitter)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


# --------------------------------------------------------------------------- #
# Default extractor: LangChain LLMGraphTransformer (paper Section 4.1)
# --------------------------------------------------------------------------- #
def langchain_extractor(model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> Extractor:
    """Build an extractor backed by LangChain's ``LLMGraphTransformer``."""
    key = require_api_key(api_key)
    from langchain_core.documents import Document
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0, api_key=key)
    transformer = LLMGraphTransformer(llm=llm)

    def _extract(text: str) -> Tuple[List[Node], List[Relationship]]:
        graph_docs = transformer.convert_to_graph_documents([Document(page_content=text)])
        nodes: List[Node] = []
        rels: List[Relationship] = []
        for gd in graph_docs:
            for n in gd.nodes:
                nodes.append(Node(id=str(n.id), type=str(n.type), properties=dict(n.properties or {})))
            for r in gd.relationships:
                rels.append(Relationship(source=str(r.source.id), target=str(r.target.id), type=str(r.type)))
        return nodes, rels

    return _extract


# --------------------------------------------------------------------------- #
# Build + persist
# --------------------------------------------------------------------------- #
def build_graph(
    source,
    out_path,
    *,
    model: str = DEFAULT_MODEL,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    extractor: Optional[Extractor] = None,
    api_key: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Build the CausalRAG text graph and write it to ``out_path`` (a .json file).

    Returns the path to the written graph. Pass ``extractor=`` to bypass the
    default LangChain backend (e.g. for offline tests).
    """
    text = _read_text(source)
    chunks = _chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No usable text found in source: {source!r}")
    log(f"[index] split into {len(chunks)} chunk(s)")

    if extractor is None:
        extractor = langchain_extractor(model=model, api_key=api_key)

    nodes_by_id: Dict[str, Node] = {}
    relationships: List[Relationship] = []
    seen_rel: set = set()
    for i, chunk in enumerate(chunks):
        chunk_nodes, chunk_rels = extractor(chunk)
        for n in chunk_nodes:
            if n.id and n.id not in nodes_by_id:
                nodes_by_id[n.id] = n
        for r in chunk_rels:
            key = (r.source, r.target, r.type)
            if r.source and r.target and key not in seen_rel:
                relationships.append(r)
                seen_rel.add(key)
        log(f"[index] chunk {i + 1}/{len(chunks)}: {len(nodes_by_id)} nodes, {len(relationships)} edges")

    # ensure every endpoint exists as a node
    for r in relationships:
        for nid in (r.source, r.target):
            nodes_by_id.setdefault(nid, Node(id=nid))

    payload = {
        "nodes": [{"id": n.id, "type": n.type, "properties": n.properties} for n in nodes_by_id.values()],
        "relationships": [{"source": r.source, "target": r.target, "type": r.type} for r in relationships],
    }
    out_path = Path(out_path)
    if out_path.suffix != ".json":
        out_path = out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[index] wrote graph ({len(payload['nodes'])} nodes, {len(payload['relationships'])} edges) to {out_path}")
    return out_path


def load_graph(graph_path) -> Tuple[List[Node], List[Relationship]]:
    """Load a built graph into (nodes, relationships)."""
    data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes = [Node(id=n["id"], type=n.get("type", "Entity"), properties=n.get("properties", {}))
             for n in data.get("nodes", [])]
    rels = [Relationship(source=r["source"], target=r["target"], type=r.get("type", "related"))
            for r in data.get("relationships", [])]
    return nodes, rels
