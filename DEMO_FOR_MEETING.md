# Using This Demo for the Foundation Meeting

**Main demo URL (recommended):** **http://127.0.0.1:8000/insights** — Single page that loads and shows Synthesis, Testable hypotheses, and Ranked evidence (predictions) with no button to click. Use this for the meeting.

**Meeting focus:** Catalyzing faster, more open research cycles in education (especially AI impacts) by adopting approaches from drug discovery and foundational AI—simulation, rapid feedback loops, open knowledge sharing—to compress the timeline from research question to actionable insight for grantmaking and partnerships. The session includes a **demonstration of an early-stage Insights for AI in Education engine prototype** that uses AI to synthesize research findings and generate testable hypotheses. The goal is to get direction on investment in experimental research infrastructure and risk tolerance for AI-accelerated approaches to building education evidence.

---

## How This Demo Responds to the Meeting

| Meeting theme | How this demo responds |
|---------------|------------------------|
| **Faster, more open research cycles** | The pipeline is **open and reusable**: citation data + a published model (CCM/LTCM). Same approach can be run on education-specific citation datasets (e.g. ERIC, education journals) to surface high-impact and emerging work quickly. |
| **AI impacts in education** | The **education topic mapping** (Policy & Systems, Learning Sciences & Instruction, Assessment & Evaluation) shows how we can **tag and rank literature by domain**. With real education + AI literature, we can identify which topics are gaining traction and where evidence is concentrated. |
| **Approaches from other fields** | The underlying **Community Citation Model** comes from science-of-science and has been used in physics (APS), law, and patents. We are **reusing that infrastructure** for education—same idea as borrowing rapid cycles from drug discovery. |
| **Synthesize research findings** | The demo **synthesizes the citation network** into a ranked list of papers by (predicted or current) impact, with topic labels. With an LLM or summarization layer, the next step is to turn “top papers in topic X” into **short synthesis bullets** (e.g. “Key finding 1… Key finding 2…”). |
| **Generate testable hypotheses** | From “top papers” and topics we can prompt an AI to propose **testable hypotheses** (e.g. “If X is true in Learning Sciences, we would expect Y in assessment outcomes”). The current demo provides the **evidence base** (which papers, which topics); the hypothesis layer can sit on top. |
| **Inform grantmaking and partnerships** | Ranked, topic-labeled evidence helps prioritize **where the evidence is strong vs. thin**. That can inform where to fund replications, where to build partnerships, and where to run simulations or pilots. |
| **Experimental research infrastructure / risk tolerance** | This is a **concrete example of experimental infrastructure**: open data + open model + small API + chat/viewer. It is low-cost, modular, and can be extended (better data, LLM synthesis, hypothesis generation) to test how much AI-acceleration is useful before scaling. |

---

## Positioning as “Insights for AI in Education Engine (Prototype)”

You can describe the demo as:

- **“An early-stage prototype of an Insights for AI in Education engine.”**
- It uses **citation and topic structure** (today: demo data with education-themed topics; tomorrow: real education/AI literature) to:
  - **Surface** high-impact and emerging papers.
  - **Organize** evidence by education-relevant topics.
  - **Prepare the ground** for AI to synthesize findings and generate testable hypotheses (next step).

**Talking points:**

1. **“We’re reusing infrastructure from other fields.”**  
   Same family of models used in physics and patents, now applied to an education-relevant structure (topics, impact ranking).

2. **“This compresses the ‘find the evidence’ step.”**  
   Instead of manual search and sifting, we get a ranked, topic-labeled list that can feed synthesis and hypothesis generation.

3. **“It’s open and extensible.”**  
   Open data formats, open model, small API. We can plug in better education datasets and add an AI layer for synthesis and hypotheses without rebuilding from scratch.

4. **“It’s an experiment we can scale or stop.”**  
   Low-cost prototype to test appetite for AI-accelerated evidence infrastructure and to get direction on investment and risk.

---

## Optional Enhancements to Better Match the Meeting

To make the demo feel even more like “synthesize findings + generate hypotheses,” you could add (conceptually or in code):

1. **Synthesis step**  
   When the user asks for “top papers” or “predict citations,” the API (or a separate service) calls an LLM with the list of titles/topics and returns 3–5 short **synthesis bullets** (e.g. “Current evidence in Learning Sciences & Instruction suggests…”).

2. **Hypothesis step**  
   From the same list (or from synthesis), prompt an LLM to output 2–3 **testable hypotheses** (e.g. “Hypothesis 1: If structured feedback is used in X context, we would see Y in outcomes. Testable by…”).

3. **One-click “Synthesize” / “Generate hypotheses” in the UI**  
   Buttons that call these steps and show the result in the chat or on the view page.

**What you can show live (no LLM required):** In the chat or API, try **“predict citations”**, **“synthesize”**, and **“hypotheses”** (or **“testable”**). The demo returns ranked evidence plus short narrative text for synthesis and testable hypotheses, so you can run it as-is in the meeting.

---

## Summary

This demo **responds to the meeting** by showing a working, open, education-topic-aware evidence pipeline that (1) ranks literature by impact, (2) tags it by education topics, and (3) can be extended to synthesize findings and generate testable hypotheses. It illustrates the kind of **experimental, AI-enabled research infrastructure** the meeting is about and supports a conversation on investment and risk tolerance.
