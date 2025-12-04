# veira

This repository contains the code used in the paper "Pandemic-Potential Viruses are a Blind Spot for Frontier Open-Source LLMs."

---

### Table of Contents
1. [Directory Structure](#directory-structure)
2. [What Each File/Folder Contains](#what-each-filefolder-contains)
3. [Inference](#inference)
   - [Step 1: Environment Setup](#step-1-environment-setup)
   - [Step 2: Modify the Config](#step-2-modify-the-config)
   - [Step 3: Run Inference](#step-3-run-inference)

---   

### Directory structure

```text
└── veira/
    ├── README.md
    ├── LICENSE
    ├── configs/
    │   └── neurips_config.yaml
    ├── data/
    │   └── dataset.py
    ├── utils/
    │   ├── prompts.py
    │   ├── utils.py
    │   └── create_vignette.py
    ├── notebooks/
    │   ├── 1_data_cleanup.ipynb
    │   ├── 2_impute_missing.ipynb
    │   ├── 3_viral-vs-non-viral-def.ipynb
    │   ├── 4_test-train-split.ipynb
    │   ├── rag_build_vectorstore_sentinel.ipynb
    │   ├── run_rf_model.ipynb
    │   └── plot_prediction_results.ipynb
    └── src/
        ├── build_rag_vectorstore.py
        ├── llm_generate_training_summary.py
        ├── llm_system_prompt.py
        ├── model_llm.py
        ├── model_llm_rag_funcs.py
        ├── model_rf.py
        ├── run_gemma_inference.py
        └── utils.py
```

### What each file/folder contains

#### `notebooks/`

* `1_data_cleanup.ipynb`  
  Cleans raw clinical data, removes invalid IDs/columns, selects relevant features, derives per‑pathogen labels, and saves a cleaned dataset with basic exploratory plots.
* `2_impute_missing.ipynb`   
  Performs numeric/categorical imputation, visualizes missingness, compares original vs. imputed distributions, and saves the imputed dataset.
* `3_viral-vs-non-viral-def.ipynb`  
  Creates a composite `all-viral_label` (viral vs. non‑viral) using existing diagnoses, defines high‑confidence negatives, and marks unknowns.
* `4_test-train-split.ipynb`  
  Builds stratified train/test splits for the `all-viral_label` task and saves reproducible pickled splits.
* `rag_build_vectorstore_sentinel.ipynb`  
  Converts per‑patient rows to text, embeds them, and builds a vector store to enable retrieval‑augmented evaluation.
* `run_rf_model.ipynb`  
  Trains and evaluates a random‑forest baseline for viral vs. non‑viral prediction; produces ROC/confusion plots, feature importances, ablations, and prediction exports.
* `plot_prediction_results.ipynb`  
  Visualizes and compares prediction outputs from all models and hybrid approaches (e.g., probability distributions, sequencing priority summaries).
  

#### `src/`

* `build_rag_vectorstore.py`  
  Turns a DataFrame into a searchable vector store (row‑to‑JSON text, embeddings, index build, save/load utilities).
* `llm_generate_training_summary.py`  
  Generates a concise knowledge summary from labeled training cases using token‑aware batching and merging.
* `llm_system_prompt.py`  
  Cleans field definitions from the data dictionary and defines reusable system prompts for model runs.
* `model_llm.py`  
  Driver for LLM‑based predictions with optional knowledge summaries and retrieval.
* `model_llm_rag_funcs.py`  
  Helper functions to load the vector store, embed queries, retrieve nearest neighbors, and format retrieved context.
* `model_rf.py`  
  End‑to‑end random‑forest pipeline: preprocessing, training/evaluation, ROC and confusion matrices, feature importances, and feature‑set ablations.
* `run_gemma_inference.py`  
  Code to pull and run inference on the RL-tuned gemma-3-4b-it instance described in the paper. To run it see inference section. 
* `utils.py`  
  Shared helpers: feature selection (`data_to_use`), column dictionary lookups, row‑to‑JSON serialization, and approximate token length utilities.

#### `configs/`

* `neurips_config.yaml`  
  Config file for load and running RL-tuned Gemma4B model presented at neurips, use this config to get same results

#### `utils/`

* `create_vignette.py`  
  Helper functions for generating patient vignette that is ultimately fed to Gemma4B.
* `prompts.py`  
  Stores system prompts used for Gemma4B model.
* `utils.py`  
  Util functions used to parse Gemma4B outputs and prepare Gemma4B inputs.

---

### Inference

This section describes how to reproduce the Gemma-3-4B-IT inference results reported in the paper, using the RL-tuned checkpoint released in this repository.


#### Step 1: Environment Setup

Create a Python environment with PyTorch and Hugging Face Transformers. Nothing beyond a standard GPU-enabled setup is required.

You must also install the Hugging Face CLI and authenticate so you can access the RL-tuned Gemma-3-4B-IT weights:

```bash
pip install transformers accelerate huggingface_hub
huggingface-cli login
```

Ensure your account has permission to pull the following directories on huggingface:

[Sentinel-AI/neurips_grpo_model](https://huggingface.co/Sentinel-AI/neurips_grpo_model)

[google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

#### Step 2: Modify the config
Edit configs/neurips_config.yaml to point to your local data and desired runtime settings.
The following fields typically require updating:

- Data section
  - train_test_data_folder
  - pathogen
  - data_path
  - col_meta_path
  - sheet_name
  - data_dictionary_path

- Model section
  - cache_dir — where model weights will be stored locally

- Eval section
  - eval_name — name for this evaluation run
  - device — e.g., "cuda:0" or "cuda:1"

Make sure all paths exist before running the inference script.

#### Step 3: Run inference

From the repository root, run:

```bash
python -m src.run_gemma_inference configs/neurips_config.yaml
```

The script will:

1. Load the RL-tuned Gemma model
2. Convert each patient row into a vignette
3. Run inference over the full test set
4. Save model probabilities and parsed outputs

Typical runtime is ~2 minutes on a single modern GPU (A100 or H100 but can work on smaller GPUs).

---

Fun fact: “Veira” means “virus” in Icelandic (h/t Kristján Eldjárn Hjörleifsson)
