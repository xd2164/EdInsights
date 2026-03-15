# How the topics in education (demo groups) are identified

## In the demo data

In the **demo** dataset (`repro/data/demo/preprocessed/`), the “topics” are **only numeric labels**:

- **`paper_table.csv`** has a column **`group`** with values **0, 1, or 2**. Each paper is assigned one of these groups.
- **`category_table.csv`** lists those same groups and a **title** for each. In the demo, the titles are literally **"0"**, **"1"**, **"2"** (no real topic names).

So in the demo, **groups are identified only as numbers**. There is no semantic label (like “Education” or “Physics”) stored in the demo files. The demo is synthetic: it was generated for testing the CCM pipeline, not from real education or discipline data.

---

## Where the education topic names come from

The three **education-themed topic names** used in this project:

| Group | Topic name we use |
|-------|-------------------|
| 0 | Education Policy & Systems |
| 1 | Learning Sciences & Instruction |
| 2 | Assessment & Evaluation |

are **not** read from the data. They were **chosen for this demo** to give the numeric groups an education interpretation. So:

- **Identified from data:** only the **group id** (0, 1, 2) from `paper_table.csv` and `category_table.csv`.
- **Assigned by us:** the **topic names** (e.g. “Education Policy & Systems”) and their short descriptions, for display and explanation in the API and docs.

If you use different labels (e.g. “Topic A”, “Topic B”, “Topic C”, or other education areas), you would change the mapping in the API (e.g. `EDUCATION_TOPICS` in `api/main.py`) or in your own code; the underlying data stays the same.

---

## How topics are identified in real CCM datasets

In **real** CCM datasets used in the repo (e.g. APS, Case Law, USPTO), topics/categories **are** identified from the **data source** and stored in the preprocessed files:

1. **Source taxonomy**  
   The provider (e.g. a publisher or database) assigns each paper to one or more categories (e.g. journal subject codes, PACS codes, patent IPC classes). Those codes or labels define the “topics.”

2. **Preprocessed files**  
   When building the CCM input:
   - **`paper_table.csv`** gets a **`group`** (or similar) that comes from that taxonomy (e.g. mapped from PACS to a small set of group ids).
   - **`category_table.csv`** gets a **`title`** (and optionally other fields) that are the **real category names** from the source (e.g. “Nuclear physics”, “Condensed matter”). See e.g. `repro/data_suppl/aps-category-name.csv` for how APS physics categories are named.

3. **Optional: your own taxonomy**  
   For education (or any domain), you could:
   - Use an existing taxonomy (e.g. ERIC descriptors, field-of-study codes) and assign each paper a **group** and a **title** when building `paper_table.csv` and `category_table.csv`.
   - Or derive groups from keywords, abstracts, or classifiers, then store the resulting group id and a human-readable topic name in the same way.

So: **in the demo, topic “identification” is only the numeric group (0/1/2); the education topic names are our chosen labels. In real data, topics are identified by the source taxonomy (or your own) and stored as proper category names in the preprocessed tables.**
