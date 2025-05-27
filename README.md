# TOPOL: Transformer-Based Narrative Polarity Fields

TOPOL is a computational framework for detecting and explaining multidimensional semantic polarity shifts in text corpora. It models discourse evolution as vector displacements in transformer embedding space, enabling unsupervised reconstruction of polarity vector fields. The framework supports both institutionally constrained texts (e.g., central bank speeches) and high-variance sentiment corpora (e.g., Amazon product reviews).

---

## 🚀 Quick Start

### 1. Create the environment and install dependencies

Launch the setup script:

```bash
source .devenv/setup.bashrc
```

This will:
- Create a new Conda environment named `topol`
- Install all required Python dependencies

---

### 2. Set up API keys

- Copy the example environment file:

```bash
cp .env.example .env
```

- Open `.env` and insert your API keys (e.g., OpenAI key for embedding/sentiment scoring)

---

### 3. Download Preprocessed Data (Recommended)

To avoid recomputing text cleaning, transformer embeddings, and sentiment labels, download the preprocessed data folder:

Data available at:  
https://osf.io/nr94j/?view_only=de5b6b40ada34c6ab5cccfaf22dd5d78

Unzip and place the `data/` folder at the project root.

---

## 📁 Project Structure

- `.devenv/` — Environment setup scripts
- `src/` — Core implementation (embedding, reduction, clustering, drift, interpretation)
- `notebooks/` — Experimental notebooks
- `data/` — Preprocessed data (optional download)
- `outputs/` — Analysis outputs.
- `.env.example` — Template for API key configuration

---

## 📌 Notes

- UMAP is fit on specific contextual boundaries (e.g., full negative reviews or pre-crisis speeches) to anchor the semantic space, wether...
- Leiden clustering is applied to the full UMAP graph to detect mixed-regime semantic clusters
- Drift is measured via centroid displacement per cluster
- Interpretation combines TF-IDF and KeyBERT (MMR-based reranking)
- Narrative dimensions...

---
