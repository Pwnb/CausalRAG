# Method overview

CausalRAG answers a question over an entity-level text graph
(`causalrag/method.py::query_causal_rag`), following the paper:

1. **Seed retrieval (Sec. 4.2).** The query is embedded with `all-MiniLM-L6-v2`
   and matched against the graph nodes via a FAISS `IndexFlatL2`; the top-`k`
   nodes become the seeds.

2. **s-hop expansion (Sec. 4.2).** Starting from the seeds, the retriever walks
   `s` hops along the graph's relationships, collecting the traversed
   cause -> effect edges into a causal subgraph.

3. **Causal-discovery report (Sec. 4.2).** The subgraph is rendered as
   "Network Data" and an LLM turns it into a structured causality analysis report
   using the **Causal Discovery Prompt** (Appendix A.1 / Figure 7).

4. **Grounded answer (Sec. 4.3).** A second LLM call merges that report with the
   query into the final answer using the **Causal Summary Prompt**
   (Appendix A.2 / Figure 7).

The return value is a dict with `answer`, `causal_report`, `network_data`,
`seed_nodes`, and a `meta` block (`num_relationships`, `k`, `s`, `model`).

## Key parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `model` | discovery / summary LLM | `gpt-4o-mini` |
| `embed_model` | sentence-transformers model for retrieval | `all-MiniLM-L6-v2` |
| `k` | number of initial seed nodes | `3` |
| `s` | number of expansion hops | `3` |
| `response_type` | target length/format for the answer | `Multiple Paragraphs` |

Both `model` and `embed_model` match the paper's Appendix C. `k = s = 3` is the
main experimental setting; the parameter study sweeps `k, s ∈ {1, ..., 5}`.
