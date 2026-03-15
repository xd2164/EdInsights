"""
Literature prediction API powered by Community Citation Model (CCM).
Uses demo data and LTCM from repro/libs/geocitmodel.

Run from repo root:
  pip install -r api/requirements.txt
  pip install -e repro/libs/geocitmodel
  python -m uvicorn api.main:app --reload --port 8000

Then run the frontend (community_citation_model/frontend) and open http://localhost:5174.
"""

from pathlib import Path
import sys

# Add repro so we can import geocitmodel
REPO_ROOT = Path(__file__).resolve().parent.parent
REPRO = REPO_ROOT / "repro"
sys.path.insert(0, str(REPRO))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from scipy import sparse

app = FastAPI(title="CCM Literature Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo data paths (relative to repo root)
DATA_DIR = REPO_ROOT / "repro" / "data" / "demo" / "preprocessed"
PAPER_TABLE_FILE = DATA_DIR / "paper_table.csv"
CITATION_NET_FILE = DATA_DIR / "citation_net.npz"

# Education-themed topic labels for demo groups (for display and "topics in education")
EDUCATION_TOPICS = {
    0: "Education Policy & Systems",
    1: "Learning Sciences & Instruction",
    2: "Assessment & Evaluation",
}

# In-memory cache after first load
_paper_table = None
_net = None
_ltcm_fitted = None
_pred_cache = None


def load_demo_data():
    global _paper_table, _net
    if _paper_table is not None:
        return _paper_table, _net
    if not PAPER_TABLE_FILE.exists() or not CITATION_NET_FILE.exists():
        return None, None
    _paper_table = pd.read_csv(PAPER_TABLE_FILE)
    _net = sparse.load_npz(CITATION_NET_FILE)
    return _paper_table, _net


def get_ltcm_predictions():
    """Fit LTCM on demo data (train up to t_train) and return predicted citation counts per paper."""
    global _ltcm_fitted, _pred_cache
    if _pred_cache is not None:
        return _pred_cache
    paper_table, net = load_demo_data()
    if paper_table is None or net is None:
        return None
    try:
        from geocitmodel.LTCM import LongTermCitationModel
    except Exception:
        return None
    t_pub = paper_table["year"].values.astype(np.float64)
    t_train = int(np.percentile(t_pub, 70))
    train_idx = np.where(t_pub <= t_train)[0]
    test_idx = np.where(t_pub > t_train)[0]
    if len(train_idx) < 10 or len(test_idx) < 5:
        return None
    train_net = net[train_idx, :][:, train_idx]
    t_pub_train = t_pub[train_idx]
    t_pub_test = t_pub[test_idx]
    t_max = int(np.nanmax(t_pub))
    try:
        model = LongTermCitationModel(min_ct=5, device="cpu")
        model.fit(train_net, t_pub_train, n_epochs=15, batch_size=min(10000, max(1000, train_net.nnz)))
    except Exception:
        return None
    try:
        pred_net, _ = model.predict(
        train_net,
        t_pub_train=t_pub_train,
        t_pub_test=t_pub_test,
        t_pred_min=t_train + 1,
        t_pred_max=t_max,
        m_m=30,
        )
    except Exception:
        return None
    pred_citations = np.array(pred_net.sum(axis=0)).reshape(-1)
    node_idx = np.concatenate([train_idx, test_idx])
    paper_ids = paper_table["paper_id"].values
    years = paper_table["year"].values
    groups = paper_table["group"].values
    out = []
    for i in range(len(pred_citations)):
        nid = node_idx[i] if i < len(node_idx) else i
        out.append({
            "paper_id": int(paper_ids[nid]),
            "year": int(years[nid]),
            "group": int(groups[nid]),
            "score": float(pred_citations[i]),
        })
    out.sort(key=lambda x: -x["score"])
    _pred_cache = out
    return out


def get_fallback_predictions():
    """When LTCM is not available, rank papers by current citation count (in-degree)."""
    paper_table, net = load_demo_data()
    if paper_table is None or net is None:
        return None
    in_degree = np.array(net.sum(axis=0)).reshape(-1)
    paper_ids = paper_table["paper_id"].values
    years = paper_table["year"].values
    groups = paper_table["group"].values
    out = []
    for i in range(len(in_degree)):
        out.append({
            "paper_id": int(paper_ids[i]),
            "year": int(years[i]),
            "group": int(groups[i]),
            "score": float(in_degree[i]),
        })
    out.sort(key=lambda x: -x["score"])
    return out


class ChatBody(BaseModel):
    message: str
    history: Optional[list] = []


@app.post("/api/chat")
def chat(body: ChatBody):
    message = (body.message or "").strip().lower()
    predictions = []
    pred_list = get_ltcm_predictions()
    used_ltcm = pred_list is not None
    if pred_list is None:
        pred_list = get_fallback_predictions()
    source = "LTCM" if used_ltcm else "current citations (install geocitmodel for LTCM predictions)"

    if "synthesize" in message or "synthesis" in message:
        pred_list = get_ltcm_predictions() or get_fallback_predictions()
        if pred_list:
            for p in pred_list[:10]:
                topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
                predictions.append({"title": f"Paper {p['paper_id']} ({topic})", "year": str(p["year"]), "score": round(p["score"], 2), "topic": topic, "url": "#"})
            reply = (
                "**Synthesis (demo):** Based on the current evidence base (ranked by citation impact), "
                "key themes appear in Education Policy & Systems (governance, system design), "
                "Learning Sciences & Instruction (how learning and instruction interact), and "
                "Assessment & Evaluation (outcomes measurement). A full AI-powered synthesis would summarize "
                "findings from these papers into actionable bullets for grantmaking. This prototype shows the "
                "evidence pipeline that would feed that synthesis."
            )
        else:
            reply = "Load demo data to see a synthesis-ready evidence list."
        return {"reply": reply, "predictions": predictions}

    if "hypothes" in message or "testable" in message:
        pred_list = get_ltcm_predictions() or get_fallback_predictions()
        if pred_list:
            for p in pred_list[:5]:
                topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
                predictions.append({"title": f"Paper {p['paper_id']} ({topic})", "year": str(p["year"]), "topic": topic, "url": "#"})
            reply = (
                "**Testable hypotheses (demo):** (1) Papers with higher predicted impact in Learning Sciences & Instruction "
                "will show stronger alignment with instructional-design principles when manually reviewed. (2) Topics with "
                "thinner evidence (fewer high-impact papers) are promising candidates for new grants or partnerships. "
                "A full prototype would use AI to generate such hypotheses from the ranked evidence list; this demo "
                "shows the evidence base that would drive that step."
            )
        else:
            reply = "Load demo data to generate hypothesis-ready evidence."
        return {"reply": reply, "predictions": predictions}

    if "predict" in message or "citation" in message or "literature" in message or "paper" in message or "top" in message:
        if pred_list:
            for p in pred_list[:15]:
                topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
                predictions.append({
                    "title": f"Paper {p['paper_id']} (year {p['year']})",
                    "authors": "CCM demo",
                    "year": str(p["year"]),
                    "score": round(p["score"], 2),
                    "url": "#",
                    "topic": topic,
                })
            reply = f"Using the Community Citation Model demo ({source}), here are papers ranked by predicted future citations."
        else:
            reply = "Demo data not found. Ensure repro/data/demo/preprocessed/ has paper_table.csv and citation_net.npz."
    else:
        if pred_list:
            for p in pred_list[:8]:
                topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
                predictions.append({
                    "title": f"Paper {p['paper_id']} (year {p['year']})",
                    "year": str(p["year"]),
                    "score": round(p["score"], 2),
                    "url": "#",
                    "topic": topic,
                })
            reply = f"I'm the literature prediction chatbot powered by the CCM demo ({source}). Ask for 'citation predictions' or 'top papers' to see more."
        else:
            reply = "I'm the CCM literature chatbot. Ensure demo data exists (repro/data/demo/preprocessed/), then ask for 'citation predictions' or 'top papers'."
    return {"reply": reply, "predictions": predictions}


@app.get("/api/predictions")
def predictions(q: str = ""):
    pred_list = get_ltcm_predictions() or get_fallback_predictions()
    if not pred_list:
        return {"predictions": []}
    n = min(20, max(5, len(pred_list)))
    out = []
    for p in pred_list[:n]:
        topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
        out.append({
            "title": f"Paper {p['paper_id']} (year {p['year']})",
            "year": str(p["year"]),
            "score": round(p["score"], 2),
            "url": "#",
            "topic": topic,
        })
    return {"predictions": out}


@app.get("/api/how-it-works")
def how_it_works():
    """Explain how prediction works and how topics map to education."""
    pred_list = get_ltcm_predictions()
    used_ltcm = pred_list is not None
    mode = "LTCM (Long-Term Citation Model)" if used_ltcm else "current citation count (fallback)"
    return {
        "how_prediction_works": {
            "summary": "Papers are ranked by predicted or current impact (citations).",
            "steps": [
                "1. Data: A citation network (who cites whom) and paper metadata (year, topic/group).",
                "2. Training (when LTCM is used): Use citations up to a cutoff year to fit a Long-Term Citation Model (parameters μ, σ, η) that models how citations accumulate over time.",
                "3. Prediction (LTCM): For each paper, predict how many citations it will receive in the future; rank papers by this predicted count.",
                "4. Fallback (when LTCM is not installed): Rank papers by current in-degree (number of citations received so far).",
                "5. Output: A list of papers ordered by (predicted or current) citation count, with topic labels for education.",
            ],
            "current_mode": mode,
            "reference": "Community Citation Model (CCM) / Long-Term Citation Model: Wang, Song & Barabási, Science 2013.",
        },
        "topics_in_education": [
            {"group_id": 0, "topic": "Education Policy & Systems", "description": "Systems, policy, and governance in education."},
            {"group_id": 1, "topic": "Learning Sciences & Instruction", "description": "How people learn and how instruction is designed."},
            {"group_id": 2, "topic": "Assessment & Evaluation", "description": "Measuring learning outcomes and evaluating programs."},
        ],
        "how_topics_are_identified": {
            "in_demo": "In the demo data, each paper has a numeric 'group' (0, 1, or 2) in paper_table.csv. The category_table.csv only stores these numbers; it does not contain real topic names.",
            "education_names": "The three education topic names (Education Policy & Systems, Learning Sciences & Instruction, Assessment & Evaluation) are not in the data. They were assigned for this demo to give the numeric groups an education interpretation.",
            "in_real_data": "In real CCM datasets (e.g. APS, USPTO), topics come from the data source taxonomy (e.g. journal subject codes, patent classes) and are stored as real category names in category_table.csv.",
        },
    }


@app.get("/api/health")
def health():
    paper_table, net = load_demo_data()
    ok = paper_table is not None and net is not None
    return {"ok": ok, "data_loaded": ok}


@app.get("/api/insights")
def insights():
    """Single payload for the Insights for AI in Education demo UI: synthesis, hypotheses, and ranked evidence (predictions)."""
    pred_list = get_ltcm_predictions() or get_fallback_predictions()
    predictions = []
    if pred_list:
        for i, p in enumerate(pred_list[:20]):
            topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
            predictions.append({
                "rank": i + 1,
                "paper_id": p["paper_id"],
                "title": f"Paper {p['paper_id']}",
                "year": int(p["year"]),
                "topic": topic,
                "score": round(p["score"], 2),
            })
    synthesis = (
        "Based on the current evidence base (ranked by citation impact), key themes appear in "
        "Education Policy & Systems (governance, system design), Learning Sciences & Instruction "
        "(how learning and instruction interact), and Assessment & Evaluation (outcomes measurement). "
        "A full AI-powered synthesis would summarize findings from these papers into actionable bullets "
        "for grantmaking. This prototype shows the evidence pipeline that would feed that synthesis."
    )
    hypotheses = [
        "Papers with higher predicted impact in Learning Sciences & Instruction will show stronger alignment with instructional-design principles when manually reviewed.",
        "Topics with thinner evidence (fewer high-impact papers) are promising candidates for new grants or partnerships.",
        "Ranked evidence by topic can inform where to fund replications and where to build partnerships for AI-in-education research.",
    ]
    return {
        "synthesis": synthesis,
        "hypotheses": hypotheses,
        "predictions": predictions,
        "source": "Community Citation Model (demo)" if pred_list else None,
    }


@app.get("/view-predictions", response_class=HTMLResponse)
def view_predictions_page():
    """Simple page to view predictions. Open http://127.0.0.1:8000/view-predictions and click Load predictions."""
    html_file = REPO_ROOT / "view_predictions.html"
    if not html_file.exists():
        return HTMLResponse("<body>view_predictions.html not found</body>", status_code=404)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/insights", response_class=HTMLResponse)
def insights_demo_page():
    """Insights for AI in Education demo UI. Open http://127.0.0.1:8000/insights. Loads synthesis, hypotheses, and ranked predictions on open."""
    html_file = REPO_ROOT / "insights_demo.html"
    if not html_file.exists():
        return HTMLResponse("<body>insights_demo.html not found</body>", status_code=404)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


def _hub_html():
    """Shared HTML for hub and root."""
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Insights for AI in Education — Hub</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 24px; }
  h1 { font-size: 1.5rem; }
  ul { line-height: 1.8; }
  a { color: #0969da; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }
</style>
</head>
<body>
  <h1>Insights for AI in Education</h1>
  <p>This stack combines citation-based insights with deep literature research.</p>
  <ul>
    <li><strong><a href="/insights">CCM Insights</a></strong> — Synthesis, hypotheses, and ranked evidence (this server).</li>
    <li><strong><a href="http://localhost:8501" target="_blank" rel="noopener">AI Education Research Assistant</a></strong> — Deep research + Neo4j (run separately; see <a href="https://github.com/alym00sa-dev/ai-in-education-research-agent" target="_blank" rel="noopener">repo</a>).</li>
    <li><a href="/docs">API docs</a></li>
  </ul>
  <p>See <code>INTEGRATION_WITH_RESEARCH_AGENT.md</code> in this repo for how to run both.</p>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root_page():
    """Serve hub at root so http://127.0.0.1:8000/ works."""
    return HTMLResponse(_hub_html())


@app.get("/hub", response_class=HTMLResponse)
def hub_page():
    """Hub: links to CCM Insights and Research Agent."""
    return HTMLResponse(_hub_html())


# Serve frontend if built
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
