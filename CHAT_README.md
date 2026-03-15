# Literature prediction chatbot (Community Citation Model)

This repo includes a **chat frontend** and an **API** that use the Community Citation Model (CCM) to answer queries and return **literature predictions** (papers ranked by predicted future citations) from the demo dataset.

## Quick start

### 1. Install backend dependencies

From the **repo root** (`community_citation_model`):

```bash
pip install -r api/requirements.txt
pip install -e repro/libs/geocitmodel
```

(If you use the full conda env from the main README, you can skip the second line after `conda activate citationdynamics`.)

### 2. Start the CCM API

From the repo root:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

The API loads `repro/data/demo/preprocessed/paper_table.csv` and `citation_net.npz`, fits the Long-Term Citation Model (LTCM) on first use, and caches predictions.

### 3. Install and run the frontend

In a **second terminal**, from the repo root:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5174**. The chat proxies `/api` to the backend on port 8000.

### 4. Use the chat

- Ask: **"predict citations"**, **"top papers"**, **"literature predictions"**, or similar.
- The bot returns a short explanation and a list of papers (demo: paper_id, year, group) ranked by **predicted future citations** from the CCM.

## Layout

| Path | Role |
|------|------|
| **api/main.py** | FastAPI app: `POST /api/chat`, `GET /api/predictions`, `GET /api/health`. Uses CCM demo data and LTCM. |
| **api/requirements.txt** | FastAPI, uvicorn, numpy, pandas, scipy. CCM code comes from `repro/libs/geocitmodel`. |
| **frontend/** | Vite + React chat UI: message list, input, prediction cards. Proxies `/api` → `http://127.0.0.1:8000`. |

## API contract

- **POST /api/chat**  
  Body: `{ "message": "user text", "history": [] }`  
  Response: `{ "reply": "…", "predictions": [ { "title", "authors?", "year", "score", "url?" } ] }`

- **GET /api/predictions?q=…**  
  Response: `{ "predictions": [ … ] }`

- **GET /api/health**  
  Response: `{ "ok": true, "data_loaded": true }` if demo data was found.

## Notes

- Predictions are from the **demo** citation network only. For full datasets (APS, Case Law, USPTO), run the Snakemake workflow first and then extend `api/main.py` to load that data and/or precomputed results.
- The first chat request that triggers a prediction may take a few seconds while LTCM fits and runs.
