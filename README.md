# veira

This repository contains the code used in the paper "Pandemic-Potential Viruses are a Blind Spot for Frontier Open-Source LLMs."

---

## Directory structure

```text
└── lauraluebbert-veira/
    ├── README.md
    ├── LICENSE
    ├── notebooks/
    │   ├── 1_data_cleanup.ipynb
    │   ├── 2_impute_missing.ipynb
    │   ├── 3_viral-vs-non-viral-def.ipynb
    │   ├── 4_test-train-split.ipynb
    │   ├── rag_build_vectorstore_sentinel.ipynb
    │   └── run_rf_model.ipynb
    └── src/
        ├── build_rag_vectorstore.py
        ├── llm_generate_training_summary.py
        ├── llm_system_prompt.py
        ├── model_llm.py
        ├── model_llm_rag_funcs.py
        ├── model_rf.py
        └── utils.py
```

## What each file/folder contains

* `README.md` — Project overview and context for the associated paper.
* `LICENSE` — MIT license.

### `notebooks/`

* `1_data_cleanup.ipynb` — Cleans raw clinical data, removes invalid IDs/columns, selects relevant features, derives per‑pathogen labels, and saves a cleaned dataset with basic exploratory plots.
* `2_impute_missing.ipynb` — Performs numeric/categorical imputation, visualizes missingness, compares original vs. imputed distributions, and saves the imputed dataset.
* `3_viral-vs-non-viral-def.ipynb` — Creates a composite `all-viral_label` (viral vs. non‑viral) using existing diagnoses, defines high‑confidence negatives, and marks unknowns.
* `4_test-train-split.ipynb` — Builds stratified train/test splits for the `all-viral_label` task and saves reproducible pickled splits.
* `rag_build_vectorstore_sentinel.ipynb` — Converts per‑patient rows to text, embeds them, and builds a vector store to enable retrieval‑augmented evaluation.
* `run_rf_model.ipynb` — Trains and evaluates a random‑forest baseline for viral vs. non‑viral prediction; produces ROC/confusion plots, feature importances, ablations, and prediction exports.

### `src/`

* `build_rag_vectorstore.py` — Turns a DataFrame into a searchable vector store (row‑to‑JSON text, embeddings, index build, save/load utilities).
* `llm_generate_training_summary.py` — Generates a concise knowledge summary from labeled training cases using token‑aware batching and merging.
* `llm_system_prompt.py` — Cleans field definitions from the data dictionary and defines reusable system prompts for model runs.
* `model_llm.py` — Driver for LLM‑based predictions with optional knowledge summaries and retrieval; parses structured outputs and saves results.
* `model_llm_rag_funcs.py` — Helper functions to load the vector store, embed queries, retrieve nearest neighbors, and format retrieved context.
* `model_rf.py` — End‑to‑end random‑forest pipeline: preprocessing, training/evaluation, ROC and confusion matrices, feature importances, and feature‑set ablations.
* `utils.py` — Shared helpers: feature selection (`data_to_use`), column dictionary lookups, row‑to‑JSON serialization, and approximate token length utilities.

---

Fun fact: “Veira” means “virus” in Icelandic (h/t Kristján Eldjárn Hjörleifsson)
