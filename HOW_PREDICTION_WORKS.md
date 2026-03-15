# How the prediction works & topics in education

This document explains how the literature prediction in this demo works and how the **topics** map to **education**.

---

## How the prediction works

1. **Data**  
   We use a **citation network**: a graph where each node is a paper and an edge *A → B* means “paper A cites paper B.” We also have **paper metadata**: publication year and a **group** (topic) for each paper.

2. **Training (when LTCM is available)**  
   We split the network by time: papers (and their citations) up to a cutoff year are used as **training** data. We fit the **Long-Term Citation Model (LTCM)** to estimate parameters (μ, σ, η) that describe how citations accumulate over time for papers.

3. **Prediction (LTCM)**  
   For each paper, the model **predicts** how many citations it will receive in the future. Papers are then **ranked** by this predicted citation count.

4. **Fallback (when LTCM is not installed)**  
   If the full model is not available, we use a simple **fallback**: rank papers by their **current** citation count (in-degree: how many papers cite them so far). You still see a ranking, but it is not a forward-looking prediction.

5. **Output**  
   The API returns a list of papers ordered by (predicted or current) citation count. Each paper is labeled with its **topic** so you can see how prediction relates to **topics in education**.

**Reference:** Long-Term Citation Model — Wang, Song & Barabási, *Science* 342 (2013), “Quantifying Long-Term Scientific Impact.” This repo implements the Community Citation Model (CCM) and related workflows.

---

## Topics in education (demo groups)

In the demo data, each paper has a **group** (0, 1, or 2) stored in `paper_table.csv`. The dataset does **not** contain real topic names—only these numbers. We **assign** the education-themed names below for this demo. For a full explanation of how topics are identified, see **HOW_TOPICS_ARE_IDENTIFIED.md**.

| Group | Topic (education) | Description |
|-------|-------------------|-------------|
| **0** | Education Policy & Systems | Systems, policy, and governance in education. |
| **1** | Learning Sciences & Instruction | How people learn and how instruction is designed. |
| **2** | Assessment & Evaluation | Measuring learning outcomes and evaluating programs. |

When you call the API or use the view page, each predicted paper is labeled with one of these topics so you can see **how prediction works** and **which education topic** each paper belongs to.

---

## How to check in the API

- **GET** `http://127.0.0.1:8000/api/how-it-works`  
  Returns a short explanation of how prediction works and the list of **topics in education** (same as above) in JSON.

- **POST** `http://127.0.0.1:8000/api/chat` with body  
  `{"message": "predict citations", "history": []}`  
  Returns ranked papers; each item includes a **topic** field (e.g. `"Learning Sciences & Instruction"`).

- **GET** `http://127.0.0.1:8000/view-predictions`  
  Open in a browser and click “Load predictions” to see the list; the table can be extended to show the **topic** column.
