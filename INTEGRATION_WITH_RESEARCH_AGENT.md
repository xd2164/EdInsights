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

## Connecting the two agents: Deep Research → CCM predictions (show the future)

Papers that the **Deep Research** agent surfaces can be **checked against the Community Citation Model** so you see both what the literature says and how the citation model predicts those papers will perform in the future.

**Flow:**

1. **Deep Research** (Research Agent) runs on a query and returns a list of papers (with citations, evidence quality, etc.).
2. Send those papers to the **CCM API**: `POST http://127.0.0.1:8000/api/check-papers` with body `{"paper_ids": [1, 2, 3, ...]}`. (In the demo, paper_ids are from the CCM dataset; in a full integration, the Research Agent would pass identifiers that the CCM can resolve.)
3. The CCM returns for each paper: **rank**, **predicted impact (score)**, and **topic**. Papers in the CCM dataset get a rank and score; others get `in_dataset: false`.
4. **Combined view:** Deep Research tells you what the literature says; CCM tells you the predicted future trajectory of those same papers in the citation network.

So: **Deep Research surfaces the papers; CCM checks the citation predictions to show the future.**

| Step | Where | What |
|------|--------|------|
| Run deep research | Research Agent (port 8501) | Get list of papers |
| Check citations / future impact | CCM API `POST /api/check-papers` | Get rank, score, topic per paper |
| Use both | Your workflow or dashboard | Literature + predicted impact |

See also `GET /api/how-it-works` on the CCM API for the `deep_research_cross_check` section.

---

## Deep integration: query paper topics from Deep Research and feed into the model (no paper IDs)

The CCM can **query papers and their topics directly from the Deep Research agent** and run them through the prediction pipeline—**no paper IDs**; everything is by title and topic.

**Setup:** Point the CCM API at the Research Agent API using the environment variable **`DEEP_RESEARCH_API_URL`** (e.g. the base URL of the Research Agent’s FastAPI, such as `http://localhost:8001` if that’s where you run the eduviz API).

**Flow:**

1. Run **Deep Research** (Research Agent) so it creates a session and stores papers in Neo4j.
2. On the **Insights** page, in **“Predictions from Deep Research”**, enter the **session ID** (or use **“List sessions”** to pick one).
3. Click **“Fetch papers & predict”**. The CCM calls the Deep Research API (`GET /api/v1/sessions/{session_id}/papers`), gets papers with **title and topic** (objective/outcome from the agent), then looks up each paper in **Semantic Scholar** for citation-based impact.
4. The result is a table of **title, authors, year, topic (from Deep Research), impact score**—no paper IDs.

**API endpoints (CCM):**

| Endpoint | Purpose |
|----------|--------|
| `GET /api/deep-research/sessions` | List sessions from the Deep Research agent (requires `DEEP_RESEARCH_API_URL`). |
| `POST /api/predict-from-deep-research` | Body: `{"session_id": "..."}` or `{"papers": [{"title": "...", "year": null, "topic": "..."}]}`. Fetches papers from Deep Research (if `session_id`) or uses the provided list; looks up each in Semantic Scholar; returns predictions (title, authors, year, topic, impact_score, url). |
| `POST /api/check-papers` | Body can be `{"papers": [{"title": "...", "topic": "..."}]}` to check by title/topic (Semantic Scholar lookup); or legacy `{"paper_ids": [1,2,3]}` for CCM demo. |

**Run CCM with Deep Research URL:**

```bash
export DEEP_RESEARCH_API_URL=http://localhost:8001   # or the URL of the Research Agent API
python -m uvicorn api.main:app --port 8000
```

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
