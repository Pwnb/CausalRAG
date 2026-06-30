"""LLM prompts for CausalRAG.

These reproduce the exact templates reported in the paper appendix
(Wang et al., Findings of ACL 2025, Appendix A / Figure 7):

  - CAUSAL_DISCOVERY_PROMPT : turns the retrieved subgraph ("Network Data")
                              into a structured causality-analysis report.
  - CAUSAL_SUMMARY_PROMPT   : merges that report with the query into the final,
                              grounded answer.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Causal-discovery prompt (Appendix A.1 / Figure 7, "Causal Discovery Prompt")
# Fills {graph_data} with the retrieved causal subgraph (entities + relations).
# --------------------------------------------------------------------------- #
CAUSAL_DISCOVERY_PROMPT = """
---Role---
You are a smart assistant that helps a human analyst to perform **causal discovery** and **impact assessment**. Your task is to analyze a **Network Data** and generate a professional report summarizing the causal effect and key insights.

--- Goal ---
Write a **structured, professional causality analysis report** that:
- **Identifies** key entities and their roles in the causality
- **Explains** the observed causal relationships and their potential impact
- **Assesses** the strength and credibility of causal claims based on available data

---Network Data---
{graph_data}

--- Report Format ---
**1. Introduction**
Briefly introduce the context and purpose of this causal analysis.
**2. Key Entities and Their Roles**
Provide an overview of the most important entities in the causal network and their relevance.
**3. Major Causal Pathways**
Describe the primary causal chains observed, emphasizing key cause-and-effect relationships.
**4. Confidence and Evidence Strength**
Assess the reliability of the causal claims, mentioning supporting data where available.
**5. Implications and Recommendations**
Discuss the potential impact of these causal relationships and suggest possible actions.

Write a **structured, analytical, and professional** report.
""".strip()


# --------------------------------------------------------------------------- #
# Causal-summary prompt (Appendix A.2 / Figure 7, "Causal Summary Prompt")
# Fills {causal_summary} with the report above, {response_type} with the target
# length/format, and {query} with the user question.
# --------------------------------------------------------------------------- #
CAUSAL_SUMMARY_PROMPT = """
---Role---
You are a helpful assistant responding to questions about data in the tables provided. You are also specializing in **causal reasoning and impact assessment**. Your task is to generate a structured response based on an extracted causal summary.

---Goal---
Generate a response of the target length and format that responds to the user's question, summarize all the Causal Summary from multiple analysts who focused on different parts of the dataset.
If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.
The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.
The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

---Causal Summary---
{causal_summary}

---Target Response Length and Format---
{response_type}

---User Query---
{query}
Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
""".strip()


# Default target length/format used to fill {response_type} in the summary prompt.
DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"
