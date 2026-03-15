# Integration: CCM Insights + AI-in-Education Research Agent

This document describes how the **Community Citation Model (CCM) Insights** prototype and the [**AI-in-Education Research Agent**](https://github.com/alym00sa-dev/ai-in-education-research-agent) are merged into one **Insights for AI in Education** stack.

---

## Two components

| Component | What it does | URL / location |
|-----------|----------------|-----------------|
| **CCM Insights** (this repo) | Citation-based ranking, synthesis narrative, testable hypotheses, education topics. Lightweight; no LLM or DB required for the demo. | **http://127.0.0.1:8000/insights** (after `uvicorn api.main:app --port 8000`) |
| **AI Education Research Agent** | Deep literature reviews via Open Deep Research (LangGraph), Neo4j knowledge graph, evidence mapping, problem-burden analysis. Uses Tavily, OpenAI/Anthropic, Neo4j. | **http://localhost:8501** (after Streamlit + LangGraph + Neo4j; see repo README) |

---

## How they complement each other

- **CCM Insights**: Fast, citation-driven view of “what’s high-impact by topic” and a structure for synthesis and hypotheses. Good for **grantmaking and partnership prioritization** and for **meeting demos** with no API keys.
- **Research Agent**: Full **deep research** on a query (e.g. “intelligent tutoring systems effectiveness”), with evidence quality, knowledge graph, and session history. Good for **comprehensive literature reviews** and **evidence-based decision making**.

Together they cover: **(1) quick evidence scan and ranking (CCM)** and **(2) in-depth research and knowledge graph (Research Agent)**.

---

## How to run both (merged workflow)

### 1. CCM Insights (this repo)

```bash
cd community_citation_model
python -m pip install -r api/requirements.txt
python -m pip install -e repro/libs/geocitmodel   # optional, for LTCM
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/hub** for the hub (links to both components) or **http://127.0.0.1:8000/insights** for the CCM demo UI.

### 2. AI-in-Education Research Agent (separate repo)

```bash
git clone https://github.com/alym00sa-dev/ai-in-education-research-agent.git
cd ai-in-education-research-agent
# Follow the repo README: Neo4j, .env, open_deep_research, research_assistant
# Start LangGraph server, then:
cd eduviz/research_assistant_viz   # or path to Streamlit app per repo layout
streamlit run app.py              # or wherever app.py lives
```

Open **http://localhost:8501** for the research assistant.

### 3. Use them together

- Start **CCM API** (port 8000) → open **/insights** for synthesis, hypotheses, and ranked evidence.
- Start **Research Agent** (port 8501) → run deep research queries and use the knowledge graph.
- The **Insights** page (this repo) includes a link to the Research Agent and its repo; you can add a link from the Research Agent UI back to **http://127.0.0.1:8000/insights** if desired.

---

## Optional: colocating the Research Agent in this repo

To keep one repo that contains both:

1. **Clone the Research Agent as a subfolder** (e.g. `research_agent/`):

   ```bash
   cd community_citation_model
   git clone https://github.com/alym00sa-dev/ai-in-education-research-agent.git research_agent
   ```

2. **Run CCM** from repo root and **Research Agent** from `research_agent/` (or the path where its Streamlit app and LangGraph config live), following the Research Agent’s README.

No code changes are required in either project for this; only paths and docs (e.g. this file and the README) need to reference the chosen layout.

---

## References

- **CCM / LTCM**: [Community Citation Model](https://github.com/skojaku/community_citation_model), Wang, Song & Barabási, *Science* 2013.
- **AI-in-Education Research Agent**: [alym00sa-dev/ai-in-education-research-agent](https://github.com/alym00sa-dev/ai-in-education-research-agent) — Open Deep Research, LangGraph, Neo4j, Streamlit.
