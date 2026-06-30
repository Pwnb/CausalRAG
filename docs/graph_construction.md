# Graph construction

`causalrag/graph.py` builds the text graph the method consumes (paper Sec. 4.1):

```
raw text (.txt / .pdf / string)
  -> chunk into character windows (with overlap)
  -> LLM entity + relationship extraction   (LangChain LLMGraphTransformer)
  -> dedupe nodes and relationships
  -> JSON graph
```

The LLM extraction follows the paper: each chunk is passed to LangChain's
`LLMGraphTransformer` (Chase, 2022), which parses entities (nodes) and
relationships (edges). The result is normalised into a small JSON schema:

```json
{
  "nodes":         [{"id": "...", "type": "...", "properties": {}}],
  "relationships": [{"source": "...", "target": "...", "type": "..."}]
}
```

## Build it

```python
from causalrag import build_graph

# from a file
build_graph("path/to/corpus.txt", out_path="runs/my_graph/graph.json")

# from a PDF
build_graph("paper.pdf", out_path="runs/paper/graph.json")

# from raw text or a list of strings
build_graph(["passage one", "passage two"], out_path="runs/demo/graph.json")
```

## Pluggable extractor

`build_graph(..., extractor=fn)` accepts any callable
`(text) -> (list[Node], list[Relationship])`, which bypasses the LangChain
backend. This is used by the offline test (`tests/test_pipeline.py`) to run the
full pipeline without an API key, and lets you swap in a different extraction
backend.
