# Deliverables — RoadAnomaly

| Requirement | File |
| --- | --- |
| **A. Technical report (CRISP-DM)** | `A_Technical_Report_CRISP_DM.md` |
| **B. Experimental notebook** | `B_Experimental_Notebook.ipynb` |

Supporting CSVs: `supporting_val_comparison.csv`, `supporting_test_comparison.csv`.

---

## B — Notebook contents (full pipeline)

Open `B_Experimental_Notebook.ipynb` **from the repo** (needs `frames/`, `src/`, `data/processed/`).

| Section | What it does |
| --- | --- |
| 1 Business | Problem, objectives, distinction vs Lis et al. |
| 2 Data mining | Inventory 60 pairs, category stats, histograms, image/mask overlay |
| 3 Cleaning | Integrity checks (pairing, labels {0,2}, non-empty anomalies, no split leakage) |
| 4 Preprocessing | Binary remap, **512×288** resize (no letterbox), split lock SHA, norm E1/E2, train-only augmentation |
| 5 Modeling | Build E0 / E1 / E2 architectures |
| 6 Evaluation | Val + test tables and bar charts |
| 7 Deployment | Streamlit `app.py` |
| 8 Limits | Honest limitations |

---

## A — Report CRISP-DM map

| Phase | Section in `A_Technical_Report_CRISP_DM.md` |
| --- | --- |
| Business Understanding | §3 |
| Data Understanding | §4 |
| Data Preparation | §5 |
| Modeling | §6 |
| Evaluation | §7 |
| Deployment | §8 |

---

## How to run the notebook

```powershell
cd C:\Users\man\Downloads\RoadAnomaly
# Open deliverable/B_Experimental_Notebook.ipynb in Jupyter / VS Code / Colab
# Kernel working directory = repo root OR deliverable/ (auto-detected)
```
