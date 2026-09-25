# Zepto Data & AI Platform

## Project Overview

This repository contains the three modules implemented for the Zepto AI/ML capstone:

1. Data Pipeline
2. Analytics Pipeline
3. Support Assistant

The project demonstrates an end-to-end workflow covering data collection, cleaning, relational storage, SQL analysis, exploratory data analysis, predictive modeling, retrieval-augmented policy assistance, API serving, and containerization.

## Repository Structure

```text
zepto-data-ai-platform/
├── README.md
├── requirements.txt
├── .gitignore
├── data_pipeline/
│   ├── README.md
│   ├── data_pipeline.py
│   ├── books_clean.csv
│   ├── books.db
│   ├── sql_queries.sql
│   └── sql_outputs.txt
├── analytics/
│   ├── README.md
│   ├── analytics_pipeline.py
│   ├── titanic.csv
│   ├── titanic_clean.csv
│   ├── correlation_matrix.csv
│   ├── top_correlations.csv
│   ├── classification_model_comparison.csv
│   ├── imbalance_comparison.csv
│   ├── random_forest_gridsearch_results.csv
│   ├── regression_results.csv
│   ├── regression_predictions.csv
│   ├── final_model_summary.csv
│   ├── sample_prediction.csv
│   ├── titanic_survival_pipeline.joblib
│   ├── tuned_random_forest.joblib
│   └── plots/
└── support_assistant/
    ├── README.md
    ├── Dockerfile
    ├── api.py
    ├── support_assistant.py
    ├── docs/
    └── chroma_db/
```

## Environment Setup

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- pandas
- numpy
- requests
- BeautifulSoup
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- joblib
- ChromaDB
- SentenceTransformers
- LangGraph
- FastAPI
- Uvicorn
- Pydantic

---

## Module 1 — Data Pipeline

Location:

```text
data_pipeline/
```

### Purpose

The data pipeline collects public book catalog data from Books to Scrape, cleans and enriches the data, stores it in a normalized SQLite database, and performs SQL and pandas analysis.

### Pipeline

```text
Books to Scrape
       ↓
Web scraping
       ↓
Raw book records
       ↓
Cleaning and type conversion
       ↓
GBP → INR enrichment
       ↓
Clean CSV
       ↓
Normalized SQLite database
       ↓
SQL + pandas analysis
```

### Key Design Decisions

- Six book categories are scraped.
- The resulting dataset contains more than the required 60 books.
- Prices are converted using the fixed project rate of `1 GBP = 105.50 INR`.
- The SQLite database uses separate `categories` and `books` tables.
- `category_id` is used as a foreign key from `books` to `categories`.
- SQL analysis is performed using `pd.read_sql`.
- A pandas merge reproduces the normalized SQL join.

### Run

From the repository root:

```bash
python data_pipeline/data_pipeline.py
```

See `data_pipeline/README.md` for the detailed implementation and outputs.

---

## Module 2 — Analytics

Location:

```text
analytics/
```

### Purpose

The analytics module profiles and cleans the Titanic dataset, performs exploratory analysis, evaluates classification models, studies class imbalance, tunes a Random Forest, performs multivariate linear regression, and saves the final modeling pipeline.

### Data Source

The source dataset is loaded through Seaborn's Titanic loader:

```python
sns.load_dataset('titanic')
```

A committed `analytics/titanic.csv` fallback is also maintained for reproducibility.

### Workflow

```text
Titanic dataset
       ↓
Profiling
       ↓
Cleaning
       ↓
EDA and visualizations
       ↓
Train/test split
       ↓
Preprocessing pipeline
       ↓
Classification models
       ↓
Imbalance analysis
       ↓
Random Forest GridSearchCV
       ↓
Multivariate Linear Regression
       ↓
Model comparison
       ↓
Saved pipeline + prediction
```

### Models and Evaluation

The module includes:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest with GridSearchCV and OOB evaluation
- Multivariate Linear Regression
- Baseline, class-balanced, and SMOTE imbalance comparisons

Classification evaluation includes accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves.

Regression evaluation includes MAE, RMSE, R², adjusted R², and residual analysis.

### Run

From the repository root:

```bash
python analytics/analytics_pipeline.py
```

See `analytics/README.md` for detailed methodology, results, interpretations, and artifacts.

---

## Module 3 — Support Assistant

Location:

```text
support_assistant/
```

### Purpose

The Support Assistant provides a local-first policy question-answering workflow using the eight required Zepto policy documents.

### Architecture

```text
User
  ↓
FastAPI /ask
  ↓
LangGraph StateGraph
  ↓
classify_intent
  ↓
+-----------------------+
|                       |
policy                general
|                       |
↓                       ↓
retrieve_and_answer    direct_answer
|                       |
↓                       ↓
ChromaDB              canned response
  ↓
Top 3 policy chunks
  ↓
Pydantic response
```

### Technology

- `all-MiniLM-L6-v2` for local embeddings
- ChromaDB for persistent vector retrieval
- LangGraph for workflow orchestration
- Pydantic for structured output
- FastAPI for serving the assistant
- Docker for containerization

### Deterministic Mock Mode

The default configuration is:

```text
MOCK_LLM=1
```

This mode is deterministic and does not require an external LLM API.

### Run API

From the repository root:

```bash
uvicorn support_assistant.api:app --host 0.0.0.0 --port 8000
```

Example request:

```json
{
  "query": "How long does delivery take?"
}
```

The response contains:

- `answer`
- `sources`
- `confidence`

See `support_assistant/README.md` for complete architecture, API, Docker, and validation documentation.

---

## Reproducibility

The repository contains source code and generated artifacts required to inspect and reproduce the completed workflows.

Important generated artifacts include:

- Clean data files
- SQLite database
- SQL outputs
- Titanic fallback dataset
- Analytics CSV outputs
- Visualization files
- Saved machine-learning pipelines
- Support Assistant policy corpus
- ChromaDB persistent vector store

The Support Assistant defaults to deterministic mock mode so that the core workflow does not depend on an external LLM API.

---

## Git Workflow

Development was performed on a feature branch:

```text
feature/capstone-development
```

The project history includes multiple commits covering the data pipeline, analytics pipeline, and subsequent project modules.

The feature branch is intended to be merged into `main` after final repository validation.

---

## Project Completion Checklist

- [x] Data Pipeline implemented
- [x] Web scraping implemented
- [x] Data cleaning implemented
- [x] Currency enrichment implemented
- [x] Normalized SQLite database implemented
- [x] SQL analysis implemented
- [x] Pandas join validation implemented
- [x] Analytics pipeline implemented
- [x] EDA and visualization artifacts generated
- [x] Classification models implemented
- [x] Imbalance analysis implemented
- [x] Random Forest tuning implemented
- [x] Multivariate linear regression implemented
- [x] Saved modeling pipeline implemented
- [x] Eight Support Assistant policy documents included
- [x] Local embeddings implemented
- [x] ChromaDB implemented
- [x] LangGraph workflow implemented
- [x] Pydantic structured output implemented
- [x] FastAPI `/ask` endpoint implemented
- [x] Dockerfile implemented
- [x] Support Assistant README implemented
- [ ] Final Git review
- [ ] Final commit
- [ ] Push final feature branch
- [ ] Merge feature branch into main

## License / Data Note

This repository is an educational capstone implementation. Public practice data and the specified policy corpus are used for the purposes of the assignment.
