"""CausalRAG quickstart: text -> causal graph -> grounded answer.

1. Put your key in a .env file (see .env.example):  OPENAI_API_KEY=sk-...
2. Run:  python examples/quickstart.py
"""

from __future__ import annotations

from causalrag import build_graph, query_causal_rag

TEXT = """
Sleep deprivation has a wide range of downstream effects on the human body.
When a person does not sleep enough, the body raises levels of the stress
hormone cortisol. Elevated cortisol increases blood pressure and promotes
inflammation throughout the body. Chronic inflammation, in turn, damages the
lining of blood vessels and accelerates the buildup of arterial plaque, which
raises the risk of heart disease.
"""

QUESTION = "How can poor sleep lead to cardiovascular damage?"


def main() -> None:
    # 4.1 build the text graph (LangChain LLM extraction).
    graph_path = build_graph(TEXT, out_path="runs/quickstart/graph.json")

    # 4.2 - 4.3 retrieve a causal subgraph, summarise it, and answer.
    result = query_causal_rag(str(graph_path), QUESTION, k=3, s=3)

    print("\n" + "=" * 70)
    print("Q:", QUESTION)
    print("A:", result["answer"])
    print("=" * 70)
    print("seed nodes        :", result["seed_nodes"])
    print("relationships used:", result["meta"]["num_relationships"])


if __name__ == "__main__":
    main()
