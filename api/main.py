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
import os
import json
import threading
import urllib.request
import urllib.error

# Add repro so we can import geocitmodel
REPO_ROOT = Path(__file__).resolve().parent.parent
REPRO = REPO_ROOT / "repro"
sys.path.insert(0, str(REPRO))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Any
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

# Deep Research agent API (optional): e.g. http://localhost:8001
DEEP_RESEARCH_API_URL = os.environ.get("DEEP_RESEARCH_API_URL", "").rstrip("/")

# In-memory cache after first load
_paper_table = None
_net = None
_ltcm_fitted = None
_pred_cache = None
_ss_papers_cache = None  # Semantic Scholar real papers for display
_ss_fetch_started = False  # Background fetch in progress


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
    _add_community_rank(out)
    _pred_cache = out
    return out


def get_fallback_predictions():
    """Community-centric fallback: rank by current citation count (in-degree) in the citation network. Same network defines the 'community'; we rank globally by impact."""
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
    # Add rank-within-community (topic) for community-centric interpretation
    _add_community_rank(out)
    return out


def _add_community_rank(pred_list):
    """Add community_rank = rank within same topic (group) by score. Modifies in place."""
    if not pred_list:
        return
    by_group = {}
    for p in pred_list:
        g = p["group"]
        if g not in by_group:
            by_group[g] = []
        by_group[g].append(p)
    for g, papers in by_group.items():
        papers.sort(key=lambda x: -x["score"])
        for r, p in enumerate(papers, start=1):
            p["community_rank"] = r


class ChatBody(BaseModel):
    message: str
    history: Optional[list] = []


class CheckPapersBody(BaseModel):
    """Papers to check: either paper_ids (CCM demo) or papers (title/topic from Deep Research)."""
    paper_ids: list[int] = []
    papers: Optional[List[dict]] = None  # [{"title": "...", "year": null, "topic": "..."}]


class PredictFromDeepResearchBody(BaseModel):
    """Either session_id (fetch papers from Deep Research API) or papers list (title/topic)."""
    session_id: Optional[str] = None
    papers: Optional[List[dict]] = None  # [{"title": "...", "year": null, "topic": "..."}]


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


def _build_paper_rank_lookup():
    """Build paper_id -> {rank, score, topic} for all papers in predictions (for check-papers)."""
    pred_list = get_ltcm_predictions() or get_fallback_predictions()
    if not pred_list:
        return None
    lookup = {}
    for rank, p in enumerate(pred_list, start=1):
        topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
        lookup[int(p["paper_id"])] = {
            "rank": rank,
            "score": round(p["score"], 2),
            "topic": topic,
        }
    return lookup


@app.get("/api/search-papers")
def search_papers(query: str = "", limit: int = 20):
    """
    Search papers by topic/keyword (e.g. tutoring, intelligent tutoring systems).
    Queries Semantic Scholar and returns papers with title, authors, year, topic, impact (citation count). Year filter: 2024 onward.
    """
    q = (query or "").strip()
    if not q:
        return {"predictions": [], "message": "Provide query (e.g. tutoring, intelligent tutoring systems)."}
    limit = min(max(1, limit), 50)
    try:
        encoded = urllib.parse.quote(q[:200])
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={encoded}&limit={limit * 2}&fields=title,year,authors,url,citationCount,fieldsOfStudy,s2FieldsOfStudy"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "EdInsights-CCM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"predictions": [], "message": f"Search failed: {e}"}
    items = data.get("data") or []
    out = []
    for item in items:
        year_val = item.get("year")
        try:
            y = int(year_val) if year_val is not None else 0
        except (TypeError, ValueError):
            y = 0
        if y < PAPER_YEAR_MIN:
            continue
        authors = item.get("authors") or []
        authors_str = ", ".join(a.get("name", "") for a in authors[:5])
        if len(authors) > 5:
            authors_str += " et al."
        topics_raw = item.get("s2FieldsOfStudy") or item.get("fieldsOfStudy") or []
        topic_names = []
        for t in (topics_raw[:3] if isinstance(topics_raw, list) else []):
            if isinstance(t, dict):
                topic_names.append(t.get("category") or t.get("displayName") or t.get("name") or "")
            elif isinstance(t, str):
                topic_names.append(t)
        topics_str = ", ".join(x for x in topic_names if x) or None
        out.append({
            "title": item.get("title") or "Untitled",
            "authors": authors_str,
            "year": item.get("year"),
            "topic": topics_str,
            "impact_score": item.get("citationCount"),
            "url": item.get("url"),
        })
        if len(out) >= limit:
            break
    out.sort(key=lambda x: (x.get("impact_score") is None, -(x.get("impact_score") or 0)))
    return {"predictions": out, "query": q, "count": len(out)}


@app.get("/api/deep-research/sessions")
def deep_research_sessions():
    """List sessions from the Deep Research agent (set DEEP_RESEARCH_API_URL)."""
    if not DEEP_RESEARCH_API_URL:
        return {"sessions": [], "message": "Set DEEP_RESEARCH_API_URL to connect to the Deep Research agent."}
    sessions = _fetch_deep_research_sessions(limit=20)
    return {"sessions": sessions, "deep_research_api": DEEP_RESEARCH_API_URL}


@app.post("/api/predict-from-deep-research")
def predict_from_deep_research(body: PredictFromDeepResearchBody):
    """
    Query paper topics from the Deep Research agent and feed into prediction.
    - Send session_id: fetch papers from Deep Research API (GET /api/v1/sessions/{id}/papers), then look up each paper in Semantic Scholar for citation-based impact. No paper IDs; use title and topic from Deep Research.
    - Or send papers: [{"title": "...", "year": null, "topic": "..."}] to run the same pipeline without calling Deep Research.
    Returns list of { title, authors, year, topic, topic_from_dr, impact_score, url }.
    """
    papers_to_predict = []
    source = None
    if body.session_id and DEEP_RESEARCH_API_URL:
        papers_to_predict = _fetch_deep_research_papers(body.session_id)
        source = f"Deep Research session {body.session_id[:8]}..."
    if not papers_to_predict and body.papers:
        papers_to_predict = [{"title": p.get("title"), "year": p.get("year"), "topic": p.get("topic")} for p in body.papers]
        source = "Provided papers list"
    if not papers_to_predict:
        return {
            "predictions": [],
            "message": "Provide session_id (and set DEEP_RESEARCH_API_URL) or papers list with title.",
        }
    predictions = _predict_papers_via_semantic_scholar(papers_to_predict)
    # Sort by impact_score descending (nulls last)
    predictions.sort(key=lambda x: (x.get("impact_score") is None, -(x.get("impact_score") or 0)))
    return {
        "predictions": predictions,
        "source": source,
        "count": len(predictions),
    }


@app.post("/api/check-papers")
def check_papers(body: CheckPapersBody):
    """
    Check papers against predictions. Two modes:
    - paper_ids: CCM demo IDs → rank/score from CCM dataset.
    - papers: [{"title": "...", "year": null, "topic": "..."}] → look up each in Semantic Scholar, return impact (citation count) and topic. No paper IDs.
    """
    # New mode: papers by title/topic (from Deep Research)
    if body.papers:
        results = _predict_papers_via_semantic_scholar(body.papers)
        return {
            "checked": len(results),
            "papers": [{"title": r["title"], "authors": r.get("authors"), "year": r.get("year"), "topic": r.get("topic"), "impact_score": r.get("impact_score"), "url": r.get("url"), "error": r.get("error")} for r in results],
            "workflow": "Papers (title/topic) → Semantic Scholar lookup → impact score. Topic from your input or from Semantic Scholar.",
        }
    # Legacy: paper_ids in CCM demo
    lookup = _build_paper_rank_lookup()
    results = []
    for pid in body.paper_ids or []:
        if lookup is None:
            results.append({"paper_id": pid, "in_dataset": False, "message": "CCM data not loaded"})
            continue
        if pid in lookup:
            results.append({
                "paper_id": pid,
                "in_dataset": True,
                "rank": lookup[pid]["rank"],
                "score": lookup[pid]["score"],
                "topic": lookup[pid]["topic"],
            })
        else:
            results.append({
                "paper_id": pid,
                "in_dataset": False,
                "message": "Paper not in CCM demo dataset.",
            })
    return {
        "checked": len(results),
        "papers": results,
        "workflow": "Deep Research returns papers → send paper_ids or papers (title/topic) here → get impact to show the future.",
    }


@app.get("/api/how-it-works")
def how_it_works():
    """Explain how prediction works and how topics map to education."""
    pred_list = get_ltcm_predictions()
    used_ltcm = pred_list is not None
    mode = "LTCM (Long-Term Citation Model)" if used_ltcm else "current citation count (fallback)"
    return {
        "how_prediction_works": {
            "summary": "Predictions are based on the Community Citation Model (CCM): the citation network is the central object; papers are ranked by predicted or current impact in that network.",
            "steps": [
                "1. Data: A citation network (who cites whom) and paper metadata (year, topic/group). The network defines the 'community' of papers and citations.",
                "2. Training (when LTCM is used): Use citations up to a cutoff year to fit a Long-Term Citation Model (LTCM) that models how citations accumulate over time.",
                "3. Prediction (LTCM): For each paper, predict future citations; rank by this predicted count (community-centric: same network, same ranking principle).",
                "4. Fallback (when LTCM is not installed): Rank papers by current in-degree in the citation network (community-centric fallback).",
                "5. Output: A list of papers ordered by (predicted or current) impact, with topic (community) labels and rank-within-topic.",
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
        "deep_research_cross_check": {
            "summary": "Papers from the Deep Research agent (AI-in-Education Research Agent) can be checked against CCM citation predictions to show future impact.",
            "flow": [
                "1. Run Deep Research on a query (e.g. intelligent tutoring systems); it returns a list of papers with citations and evidence.",
                "2. Send those paper identifiers (e.g. paper_ids) to this API: POST /api/check-papers with body {\"paper_ids\": [1, 2, 3, ...]}.",
                "3. CCM returns for each paper: rank, predicted impact (score), and topic. Papers in the CCM dataset get a rank and score; others get in_dataset: false.",
                "4. Use the combined view: Deep Research says what the literature says; CCM says how that paper is predicted to perform in the citation network—the future trajectory.",
            ],
            "endpoint": "POST /api/check-papers",
        },
    }


@app.get("/api/health")
def health():
    paper_table, net = load_demo_data()
    ok = paper_table is not None and net is not None
    return {"ok": ok, "data_loaded": ok}


@app.post("/api/clear-cache")
def clear_cache():
    """Clear in-memory caches (Semantic Scholar, predictions). Use after data changes or to force fresh fetch."""
    global _ss_papers_cache, _ss_fetch_started, _pred_cache, _ltcm_fitted
    _ss_papers_cache = None
    _ss_fetch_started = False
    _pred_cache = None
    _ltcm_fitted = None
    return {"ok": True, "message": "Cache cleared"}


# Minimum publication year for displayed papers (post 2023)
PAPER_YEAR_MIN = 2024


def _fetch_semantic_scholar_papers_sync(limit=25):
    """Sync fetch of Semantic Scholar papers. Used by background thread."""
    global _ss_papers_cache
    out = []
    queries = ["AI+in+education", "intelligent+tutoring+systems+education"]
    try:
        for q in queries:
            if len(out) >= limit:
                break
            fetch_limit = min(100, max(limit - len(out) + 30, 50))
            url = (
                "https://api.semanticscholar.org/graph/v1/paper/search"
                "?query=%s&limit=%d&fields=title,year,authors,url,fieldsOfStudy,s2FieldsOfStudy" % (q, fetch_limit)
            )
            req = urllib.request.Request(url, headers={"User-Agent": "EdInsights-CCM/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            for item in data.get("data") or []:
                if len(out) >= limit:
                    break
                year_val = item.get("year")
                try:
                    y = int(year_val) if year_val is not None else 0
                except (TypeError, ValueError):
                    y = 0
                if y < PAPER_YEAR_MIN:
                    continue
                authors = item.get("authors") or []
                authors_str = ", ".join(a.get("name", "") for a in authors[:5])
                if authors and len(authors) > 5:
                    authors_str += " et al."
                topics_raw = item.get("s2FieldsOfStudy") or item.get("fieldsOfStudy") or []
                topic_names = []
                if isinstance(topics_raw, list):
                    for t in topics_raw[:4]:
                        if isinstance(t, dict):
                            topic_names.append(t.get("category") or t.get("displayName") or t.get("name") or "")
                        elif isinstance(t, str):
                            topic_names.append(t)
                topic_names = [x for x in topic_names if x]
                topics_specific = ", ".join(topic_names) if topic_names else None
                out.append({
                    "title": item.get("title") or "Untitled",
                    "year": year_val,
                    "authors": authors_str,
                    "url": item.get("url"),
                    "topics_specific": topics_specific,
                })
        _ss_papers_cache = out[:limit]
    except Exception:
        _ss_papers_cache = []


def _fetch_semantic_scholar_papers(limit=25):
    """Return cached papers if available. Otherwise return [] and start background fetch for next request."""
    global _ss_papers_cache, _ss_fetch_started
    if _ss_papers_cache is not None:
        return _ss_papers_cache
    if not _ss_fetch_started:
        _ss_fetch_started = True
        def _bg():
            global _ss_fetch_started
            _fetch_semantic_scholar_papers_sync(limit)
            _ss_fetch_started = False
        threading.Thread(target=_bg, daemon=True).start()
    return []


def _http_get_json(url: str, timeout: int = 15) -> Any:
    """GET URL and return JSON. Raises on error or returns None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EdInsights-CCM/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _fetch_deep_research_sessions(limit: int = 10) -> List[dict]:
    """Fetch list of sessions from Deep Research API."""
    if not DEEP_RESEARCH_API_URL:
        return []
    url = f"{DEEP_RESEARCH_API_URL}/api/v1/sessions?limit={limit}"
    data = _http_get_json(url)
    if not data or "sessions" not in data:
        return []
    return data["sessions"]


def _fetch_deep_research_papers(session_id: str) -> List[dict]:
    """Fetch papers from a Deep Research session. Returns list of {title, topic_from_dr, year?}."""
    if not DEEP_RESEARCH_API_URL:
        return []
    url = f"{DEEP_RESEARCH_API_URL}/api/v1/sessions/{session_id}/papers"
    data = _http_get_json(url)
    if not data or "papers" not in data:
        return []
    out = []
    for p in data["papers"]:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        # Topic from Deep Research: objective, outcome, or other taxonomy fields
        obj = (p.get("objective") or p.get("implementation_objective") or "").strip()
        outco = (p.get("outcome") or "").strip()
        topic_parts = [x for x in [obj, outco] if x]
        topic_from_dr = " | ".join(topic_parts) if topic_parts else (p.get("population") or p.get("user_type") or "")
        out.append({"title": title, "topic": topic_from_dr, "year": p.get("year")})
    return out


def _lookup_paper_semantic_scholar(title: str, year: Optional[int] = None) -> Optional[dict]:
    """Look up a paper by title in Semantic Scholar; return first hit with citation count and metadata."""
    try:
        q = urllib.parse.quote(title[:200])
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=1&fields=title,year,authors,url,citationCount,fieldsOfStudy,s2FieldsOfStudy"
        req = urllib.request.Request(url, headers={"User-Agent": "EdInsights-CCM/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("data") or []
        if not items:
            return None
        item = items[0]
        authors = item.get("authors") or []
        authors_str = ", ".join(a.get("name", "") for a in authors[:5])
        if len(authors) > 5:
            authors_str += " et al."
        topics_raw = item.get("s2FieldsOfStudy") or item.get("fieldsOfStudy") or []
        topic_names = []
        for t in (topics_raw[:3] if isinstance(topics_raw, list) else []):
            if isinstance(t, dict):
                topic_names.append(t.get("category") or t.get("displayName") or t.get("name") or "")
            elif isinstance(t, str):
                topic_names.append(t)
        topics_str = ", ".join(x for x in topic_names if x) or None
        return {
            "title": item.get("title") or title,
            "year": item.get("year"),
            "authors": authors_str,
            "url": item.get("url"),
            "citation_count": item.get("citationCount"),
            "topics_specific": topics_str,
        }
    except Exception:
        return None


def _predict_papers_via_semantic_scholar(papers: List[dict]) -> List[dict]:
    """For each paper {title, year?, topic?}, look up in Semantic Scholar and return prediction (no paper_id)."""
    results = []
    for i, p in enumerate(papers):
        title = (p.get("title") or "").strip()
        if not title:
            results.append({"title": "", "error": "Missing title"})
            continue
        topic_from_dr = (p.get("topic") or "").strip()
        year_in = p.get("year")
        hit = _lookup_paper_semantic_scholar(title, int(year_in) if year_in is not None else None)
        if hit:
            results.append({
                "title": hit["title"],
                "authors": hit.get("authors") or "",
                "year": hit.get("year"),
                "topic": topic_from_dr or hit.get("topics_specific"),
                "topic_from_dr": topic_from_dr or None,
                "impact_score": hit.get("citation_count"),
                "url": hit.get("url"),
            })
        else:
            results.append({
                "title": title,
                "authors": "",
                "year": None,
                "topic": topic_from_dr or None,
                "topic_from_dr": topic_from_dr or None,
                "impact_score": None,
                "url": None,
                "error": "Not found in Semantic Scholar",
            })
    return results


def _topic_summary_from_predictions(pred_list, top_n=20):
    """From ranked predictions, return per-topic counts and score range for the top N."""
    if not pred_list:
        return None
    top = pred_list[:top_n]
    counts = {}
    score_by_topic = {}
    for p in top:
        g = p["group"]
        name = EDUCATION_TOPICS.get(g, f"Group {g}")
        counts[name] = counts.get(name, 0) + 1
        if name not in score_by_topic:
            score_by_topic[name] = []
        score_by_topic[name].append(p["score"])
    summary = []
    for name in EDUCATION_TOPICS.values():
        c = counts.get(name, 0)
        scores = score_by_topic.get(name, [])
        lo = round(min(scores), 1) if scores else 0
        hi = round(max(scores), 1) if scores else 0
        summary.append({"topic": name, "count": c, "score_min": lo, "score_max": hi})
    return summary


@app.get("/api/insights")
def insights():
    """Single payload for the Insights for AI in Education demo UI: synthesis, hypotheses, evidence basis, and ranked evidence (predictions). All predictions are from the Community Citation Model (CCM): the citation network is the central object; we rank by predicted or current impact."""
    pred_list = get_ltcm_predictions() or get_fallback_predictions()
    used_ltcm = get_ltcm_predictions() is not None
    model_name = "Community Citation Model (LTCM)" if used_ltcm else "Community Citation Model (in-degree)"
    predictions = []
    topic_summary = _topic_summary_from_predictions(pred_list, top_n=20) if pred_list else None

    # Real paper metadata from Semantic Scholar (AI in education) for display
    real_papers = _fetch_semantic_scholar_papers(limit=25)

    if pred_list:
        for i, p in enumerate(pred_list[:20]):
            topic = EDUCATION_TOPICS.get(p["group"], f"Group {p['group']}")
            meta = real_papers[i] if i < len(real_papers) else None
            # Only use year from real paper (post 2023); otherwise leave blank so we don't show demo years
            display_year = None
            if meta and meta.get("year") is not None:
                try:
                    y = int(meta["year"])
                    if y >= PAPER_YEAR_MIN:
                        display_year = y
                except (TypeError, ValueError):
                    pass
            # Topic: specific from paper metadata (Semantic Scholar) when available, else CCM broad topic
            topics_specific = (meta.get("topics_specific") or "").strip() if meta else ""
            topic_display = topics_specific if topics_specific else topic
            predictions.append({
                "rank": i + 1,
                "paper_id": p["paper_id"],
                "title": (meta["title"] if meta else f"Paper {p['paper_id']}"),
                "year": display_year,
                "authors": (meta.get("authors") or "") if meta else "",
                "url": (meta.get("url") or "") if meta else "",
                "topic": topic,
                "topics_specific": topics_specific or None,
                "topic_display": topic_display,
                "score": round(p["score"], 2),
                "community_rank": p.get("community_rank"),
            })

    # Synthesis: nuanced narrative tied to the central model, with caveats
    synthesis = (
        "The central model is a citation network: papers and the citations between them. We rank papers by predicted "
        "or current citation impact so the evidence base is ordered by influence as the field signals it—with the caveat "
        "that citations are a proxy for attention and use, not necessarily for quality or policy relevance. "
        "Within that frame, three thematic clusters emerge from the ranked evidence: "
        "Education Policy & Systems (governance, system design, institutional change), "
        "Learning Sciences & Instruction (how learning and instruction interact, including curriculum and pedagogy), "
        "and Assessment & Evaluation (outcomes measurement, validity, and use of data). "
        "The distribution across topics is not even; some areas concentrate more high-impact papers in the top ranks, "
        "others appear thinner—which may reflect a more diffuse literature, later emergence, or different citation norms. "
        "This prototype does not interpret findings inside each paper; it surfaces which papers the model places at the top "
        "so that a subsequent step (human or AI) can turn that list into actionable synthesis for grantmaking."
    )

    # How the evidence strengthens the synthesis: nuanced, with limits
    evidence_basis = (
        "The evidence strengthens the synthesis in three ways, each with limits. (1) The ranking comes from the "
        "citation network via the Community Citation Model (LTCM when available, or current in-citations). "
        "Papers predicted or observed to attract more citations are treated as higher-impact in this pipeline—bearing in mind "
        "that citation dynamics differ by field, time since publication, and type of work (e.g. methods vs. applications). "
        "(2) Topic labels assign each paper to one of three education themes, so we can see how impact is distributed "
        "across policy, learning/instruction, and assessment; that distribution is informative but depends on how topics "
        "were defined in the underlying data. (3) The table below is the direct output of the model—every rank and score "
        "is traceable to the data and algorithm—so grantmakers can see exactly which papers the pipeline surfaces and "
        "combine that with domain judgment (e.g. fit to priorities, feasibility of replication) rather than relying on rank alone."
    )

    # Paper-level examples: specific papers from the ranked list to make hypotheses testable per paper
    paper_level_examples = []
    if predictions and topic_summary:
        # Top papers in Learning Sciences (group 1) for H1
        learning_sciences = [p for p in predictions if EDUCATION_TOPICS.get(1) in (p.get("topic") or "")]
        if learning_sciences:
            papers_h1 = learning_sciences[:5]
            paper_list = ", ".join(f"Paper {p['paper_id']} (rank {p['rank']}, impact {p['score']})" for p in papers_h1)
            paper_level_examples.append({
                "hypothesis_ref": 1,
                "label": "Test H1 (design principles)",
                "text": f"For each of these papers from the table, run a manual review and score alignment with instructional-design principles: {paper_list}. Then correlate those scores with the impact score to test whether model-based impact tracks design quality.",
            })
        # Thinnest topic: topic with fewest papers in top 20 → specific papers for H2
        min_topic = min(topic_summary, key=lambda s: s["count"]) if topic_summary else None
        if min_topic and min_topic["count"] > 0:
            papers_in_topic = [p for p in predictions if p.get("topic") == min_topic["topic"]]
            paper_list_h2 = ", ".join(f"Paper {p['paper_id']} (rank {p['rank']})" for p in papers_in_topic[:5])
            paper_level_examples.append({
                "hypothesis_ref": 2,
                "label": "Test H2 (thinner topic → grants/partnerships)",
                "text": f"Topic «{min_topic['topic']}» has {min_topic['count']} paper(s) in the top 20: {paper_list_h2}. Treat each of these as a candidate for replication or partnership-focused follow-up; domain review can decide which are under-researched vs. differently structured.",
            })
        # Per-paper decision for H3: one example row
        if predictions:
            p = predictions[0]
            paper_level_examples.append({
                "hypothesis_ref": 3,
                "label": "Test H3 (paper-level decision)",
                "text": f"Example: Paper {p['paper_id']} (rank {p['rank']}, {p['topic']}, impact {p['score']}). Use the table row-by-row: for each paper, combine rank + topic + score with your priorities to assign a tentative action (replicate / partner / monitor), then refine with domain judgment.",
            })

    # Testable hypotheses at individual paper level (each row in the table is the unit of analysis)
    hypotheses = [
        {
            "statement": "For each paper in Learning Sciences & Instruction in the ranked table, higher predicted impact (score) will correlate with stronger alignment to instructional-design principles when that paper is manually reviewed and scored.",
            "evidence_support": "The table lists individual papers with rank, paper ID, topic, and impact score. Take the Learning Sciences rows (see ranked evidence below), select e.g. the top 5 by rank, and score each paper on design principles; then test the correlation with the impact score. A weak correlation suggests citation impact and design quality are partly decoupled for this set.",
        },
        {
            "statement": "Each paper that appears in a topic with fewer high-impact papers in the top 20 is a candidate for replication or partnership follow-up; the specific paper ID and rank in the table identify which ones.",
            "evidence_support": "From the table, filter by topic and note which topics have 1–2 papers in the top 20. Those specific papers (by paper ID and rank) are the individual candidates. Domain review then decides whether each is under-researched (good for grants/partnerships) or reflects a smaller subfield.",
        },
        {
            "statement": "For each paper in the ranked table, the combination of its rank, topic, and impact score can inform a paper-level decision: replicate (high score and topic with many high-impact papers), partner (high score but topic with few), or monitor (lower rank).",
            "evidence_support": "Every row in the table is one paper. Use rank + topic + impact score to assign a tentative action per paper (e.g. Paper 42 → replicate; Paper 17 → partner). The table is the checklist; domain judgment and strategy refine the final decision for each paper.",
        },
    ]

    # Backward compatibility: flat list of hypothesis strings
    hypotheses_flat = [h["statement"] for h in hypotheses]

    out = {
        "synthesis": synthesis,
        "evidence_basis": evidence_basis,
        "hypotheses": hypotheses_flat,
        "hypotheses_with_evidence": hypotheses,
        "paper_level_examples": paper_level_examples,
        "predictions": predictions,
        "topic_summary": topic_summary,
        "source": "Community Citation Model (demo)" if pred_list else None,
        "model_used": model_name,
        "paper_metadata_source": "Semantic Scholar (AI in education)" if real_papers else None,
    }
    return out


@app.get("/view-predictions", response_class=HTMLResponse)
def view_predictions_page():
    """Simple page to view predictions. Open http://127.0.0.1:8000/view-predictions and click Load predictions."""
    html_file = REPO_ROOT / "view_predictions.html"
    if not html_file.exists():
        return HTMLResponse("<body>view_predictions.html not found</body>", status_code=404)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


def _escape_html(s: str) -> str:
    """Escape HTML special chars."""
    if not s:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_insights_html(data: dict) -> dict:
    """Build HTML snippets for server-side rendering."""
    synthesis = _escape_html(data.get("synthesis") or "—")
    evidence_basis = _escape_html(data.get("evidence_basis") or "—")
    topic_summary = ""
    if data.get("topic_summary"):
        parts = [
            f"{s['topic']}: {s['count']} paper(s), score {s['score_min']}–{s['score_max']}"
            for s in data["topic_summary"]
        ]
        topic_summary = '<div style="display:block; font-size:0.875rem; color:var(--muted); margin-top:8px;">Top 20 by topic: ' + _escape_html("; ".join(parts)) + "</div>"
    hypotheses_html = ""
    for h in data.get("hypotheses_with_evidence") or []:
        stmt = _escape_html(h.get("statement") or "")
        ev = h.get("evidence_support") or ""
        if ev:
            hypotheses_html += f'<li>{stmt}<div class="hyp-ev">Evidence: {_escape_html(ev)}</div></li>'
        else:
            hypotheses_html += f"<li>{stmt}</li>"
    if not hypotheses_html and data.get("hypotheses"):
        for s in data["hypotheses"]:
            hypotheses_html += f"<li>{_escape_html(s)}</li>"
    predictions_html = ""
    for p in data.get("predictions") or []:
        tc = "tag-0" if "Policy" in str(p.get("topic") or "") else "tag-1" if "Learning" in str(p.get("topic") or "") else "tag-2"
        title = p.get("title") or f"Paper {p.get('paper_id', '')}"
        if p.get("url"):
            title_cell = f'<a href="{_escape_html(p["url"])}" target="_blank" rel="noopener" class="paper-link">{_escape_html(title)}</a>'
        else:
            title_cell = _escape_html(title)
        authors = _escape_html((p.get("authors") or "").strip() or "—")
        comm_rank = p.get("community_rank") if p.get("community_rank") not in (None, "") else "—"
        year_display = p.get("year") if p.get("year") not in (None, "") else "—"
        topic_display = _escape_html((p.get("topic_display") or p.get("topics_specific") or p.get("topic") or "—").strip())
        score = p.get("score") if p.get("score") is not None else ""
        predictions_html += f'<tr><td class="num">{p.get("rank", "")}</td><td>{title_cell}</td><td class="authors">{authors}</td><td>{year_display}</td><td><span class="tag {tc}">{topic_display}</span></td><td class="num">{comm_rank}</td><td class="num score">{score}</td></tr>'
    source_note = ""
    if data.get("source") or data.get("model_used") or data.get("paper_metadata_source"):
        source_note = "Source: " + " • ".join(filter(None, [data.get("source"), data.get("model_used"), data.get("paper_metadata_source")]))
    preds = data.get("predictions") or []
    top_ids = ", ".join(str(p.get("paper_id", "")) for p in preds[:3])
    return {
        "SYNTHESIS": synthesis,
        "EVIDENCE_BASIS": evidence_basis,
        "TOPIC_SUMMARY": topic_summary,
        "HYPOTHESES_HTML": hypotheses_html,
        "PREDICTIONS_HTML": predictions_html,
        "SOURCE_NOTE": _escape_html(source_note),
        "TOP_PAPER_IDS": top_ids,
    }


@app.get("/insights", response_class=HTMLResponse)
def insights_demo_page():
    """Insights for AI in Education demo UI. Content is server-rendered so it displays immediately."""
    html_file = REPO_ROOT / "insights_demo.html"
    if not html_file.exists():
        return HTMLResponse("<body>insights_demo.html not found</body>", status_code=404)
    html = html_file.read_text(encoding="utf-8")
    try:
        data = insights()
        snippets = _render_insights_html(data)
        for k, v in snippets.items():
            html = html.replace("{{" + k + "}}", v)
    except Exception as e:
        html = html.replace("{{SYNTHESIS}}", f"Error loading data: {_escape_html(str(e))}")
        html = html.replace("{{EVIDENCE_BASIS}}", "—")
        html = html.replace("{{TOPIC_SUMMARY}}", "")
        html = html.replace("{{HYPOTHESES_HTML}}", "")
        html = html.replace("{{PREDICTIONS_HTML}}", "")
        html = html.replace("{{SOURCE_NOTE}}", "")
        html = html.replace("{{TOP_PAPER_IDS}}", "")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


def _hub_html():
    """Hub: methodology, mechanism, and coherent storyline for Insights for AI in Education."""
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Insights for AI in Education — Methodology &amp; Hub</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; }
  body { font-family: Inter, system-ui, sans-serif; max-width: 680px; margin: 0 auto; padding: 28px 24px 48px; background: #f8fafc; color: #0f172a; line-height: 1.65; font-size: 15px; }
  h1 { font-size: 1.5rem; font-weight: 600; margin: 0 0 6px 0; letter-spacing: -0.02em; }
  .tagline { color: #64748b; font-size: 1rem; margin: 0 0 28px 0; padding-bottom: 20px; border-bottom: 1px solid #e2e8f0; }
  h2 { font-size: 1rem; font-weight: 600; margin: 24px 0 10px 0; color: #0f172a; }
  h2:first-of-type { margin-top: 0; }
  p { margin: 0 0 12px 0; color: #334155; font-size: 0.9375rem; }
  .story { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 22px; margin-bottom: 24px; }
  .story p { margin-bottom: 0; }
  .story p + p { margin-top: 10px; }
  ul.compact { margin: 0 0 16px 0; padding-left: 20px; font-size: 0.9375rem; color: #334155; }
  ul.compact li { margin-bottom: 4px; }
  .mechanism { display: grid; gap: 12px; margin: 12px 0 24px 0; }
  .mech-box { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; font-size: 0.9375rem; }
  .mech-box strong { color: #0f172a; }
  .mech-box p { margin: 4px 0 0 0; color: #64748b; font-size: 0.875rem; }
  .connect { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 14px 18px; margin: 16px 0 24px 0; font-size: 0.9375rem; }
  .connect strong { color: #0369a1; }
  .ctas { display: flex; flex-wrap: wrap; gap: 10px; margin: 24px 0 16px 0; }
  .ctas a { display: inline-block; padding: 10px 18px; background: #0ea5e9; color: #fff; text-decoration: none; border-radius: 8px; font-weight: 500; font-size: 0.9375rem; }
  .ctas a:hover { background: #0284c7; }
  .ctas a.outline { background: #fff; color: #0f172a; border: 1px solid #e2e8f0; }
  .ctas a.outline:hover { background: #f1f5f9; }
  .components { margin-top: 28px; padding-top: 20px; border-top: 1px solid #e2e8f0; }
  .components h2 { margin-bottom: 12px; }
  .components ul { list-style: none; padding: 0; margin: 0; }
  .components li { padding: 10px 0; border-bottom: 1px solid #e2e8f0; font-size: 0.9375rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  .components li:last-child { border-bottom: none; }
  .components a { color: #0ea5e9; text-decoration: none; font-weight: 500; }
  .components a:hover { text-decoration: underline; }
  .components .desc { color: #64748b; font-size: 0.875rem; font-weight: 400; }
  .foot { margin-top: 28px; font-size: 0.8125rem; color: #94a3b8; }
  .foot a { color: #0ea5e9; text-decoration: none; }
</style>
</head>
<body>
  <h1>Insights for AI in Education</h1>
  <p class="tagline">This stack combines citation-based insights with deep literature research to support evidence-driven decisions for grantmaking and partnerships.</p>

  <h2>Methodology</h2>
  <div class="story">
    <p><strong>Evidence pipeline.</strong> We start from the literature: papers, citations, and topics. The pipeline (1) ranks evidence by predicted or current citation impact so the most influential work surfaces first, (2) organizes that evidence by theme (e.g. policy, learning sciences, assessment), and (3) turns the ranked list into synthesis and testable hypotheses. The outcome is a traceable evidence base—every rank and score ties back to the citation network—so decisions can combine model output with domain judgment.</p>
    <p><strong>Two lenses.</strong> Citation-based ranking answers &ldquo;what does the field signal as important?&rdquo; Deep literature research answers &ldquo;what does the literature actually say?&rdquo; Used together, they give both influence and content; papers from deep research can be run through the citation model to see their predicted future trajectory.</p>
  </div>

  <h2>Mechanism</h2>
  <div class="mechanism">
    <div class="mech-box"><strong>Community Citation Model (CCM)</strong><p>Citation network (who cites whom) plus a Long-Term Citation Model (or current in-citations) ranks papers by predicted impact. Output: a ranked list with topic labels. This server serves synthesis, evidence basis, testable hypotheses, and the ranked table.</p></div>
    <div class="mech-box"><strong>Deep Research (Research Agent)</strong><p>Open Deep Research (LangGraph) runs full literature reviews: search, evidence quality, report generation. Writes to Neo4j for knowledge graph and evidence mapping. Run separately; returns a list of papers.</p></div>
    <div class="connect"><strong>Connection.</strong> Papers from Deep Research can be sent to this API (<code>POST /api/check-papers</code>). CCM returns rank, predicted impact, and topic for each paper in its dataset—so you see both what the literature says and how the citation model predicts those papers will perform (the future).</div>
  </div>

  <h2>Run it</h2>
  <p>Use the components below. For setup and how to run both agents together, see <code>INTEGRATION_WITH_RESEARCH_AGENT.md</code> in this repo.</p>
  <div class="ctas">
    <a href="/insights">CCM Insights</a>
    <a href="http://localhost:8501" target="_blank" rel="noopener">Deep Research</a>
    <a href="/docs" class="outline">API docs</a>
  </div>

  <div class="components">
    <h2>Components</h2>
    <ul>
      <li><a href="/insights">CCM Insights</a><span class="desc">Synthesis, hypotheses, ranked evidence, and Check papers (this server)</span></li>
      <li><a href="http://localhost:8501" target="_blank" rel="noopener">AI Education Research Assistant</a><span class="desc">Deep research + Neo4j (run separately; see repo)</span></li>
      <li><a href="/docs">API docs</a><span class="desc">REST API including <code>/api/check-papers</code></span></li>
    </ul>
  </div>

  <p class="foot"><a href="https://github.com/alym00sa-dev/ai-in-education-research-agent" target="_blank" rel="noopener">Research Agent repo</a> &middot; Integration: <code>INTEGRATION_WITH_RESEARCH_AGENT.md</code></p>
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
