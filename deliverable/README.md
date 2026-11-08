# Deliverables — RoadAnomaly

Package for the academic submission (required items A + B).

| Requirement | File in this folder |
| --- | --- |
| **A. Technical report (CRISP-DM)** | `A_Technical_Report_CRISP_DM.md` |
| **B. Experimental notebook** | `B_Experimental_Notebook.ipynb` |

Supporting tables (evaluation evidence, optional annex):

- `supporting_val_comparison.csv` — validation leaderboard E0/E1/E2a/E2b  
- `supporting_test_comparison.csv` — one-shot test results (Stage 5)

---

## A. Technical report — CRISP-DM map

Open `A_Technical_Report_CRISP_DM.md` (French). Sections:

| Required phase | Report section |
| --- | --- |
| Business Understanding | §3 |
| Data Understanding | §4 |
| Data Preparation | §5 |
| Modeling | §6 (E0, E1, E2a, E2b) |
| Evaluation | §7 (validation + test) |
| Deployment | §8 (Streamlit demo `app.py`) |

Primary model: **E2b**, operational threshold **0.55** (frozen before test).

---

## B. Jupyter / Colab notebook

Open `B_Experimental_Notebook.ipynb` in Jupyter, VS Code, or upload to Colab.

It covers, with section titles and comments:

1. Project identity / protocol  
2. Business & data understanding  
3. Data preparation  
4. Modeling (E0–E2)  
5. Validation evaluation  
6. Final test (Stage 5 results)  
7. Limits & conclusion  

Related execution notebooks in the main repo (not duplicated here):

- `notebooks/01_exploration_and_preprocessing.ipynb`  
- `notebooks/02_phase4_training_colab.ipynb`  
- `notebooks/03_final_test_evaluation_colab.ipynb`  

Deployment demo (optional for §Deployment): from repo root, `streamlit run app.py`.

---

## How to submit

1. Convert `A_Technical_Report_CRISP_DM.md` to PDF if your instructor requires PDF (e.g. Pandoc / VS Code Markdown PDF).  
2. Include `B_Experimental_Notebook.ipynb` (`.ipynb` or exported HTML/PDF).  
3. Keep this `deliverable/` folder together with the two supporting CSVs if useful as annex.
