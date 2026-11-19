# Telecom AI Brand Intelligence System

An AI-powered telecom customer feedback analysis system built around **ZENDS Communications**, a simulated telecom company. The application combines traditional NLP classification with Retrieval-Augmented Generation (RAG) to analyse customer feedback, identify service-related issues, retrieve relevant company information, and support brand analysts with dashboard insights.

> **Project note:** ZENDS Communications is a virtual company created for this project. The feedback dataset is synthetic and is used to simulate telecom customer reviews and complaints.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Technology Stack](#technology-stack)
- [Machine Learning](#machine-learning)
- [Retrieval-Augmented Generation](#retrieval-augmented-generation)
- [Streamlit Application](#streamlit-application)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Configuration and Notes](#configuration-and-notes)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

## Overview

Telecom providers receive customer feedback through channels such as surveys, support platforms, mobile application reviews, and online feedback portals. Manually reviewing large volumes of unstructured messages can make it difficult to identify customer sentiment, recurring complaints, and service issues.

This project provides a local AI-based workflow that:

1. Classifies customer feedback as **Positive**, **Negative**, or **Neutral**.
2. Predicts the telecom service category associated with the feedback.
3. Retrieves relevant information from company and telecom knowledge-base PDFs.
4. Generates a response grounded in the retrieved document context.
5. Presents feedback trends and complaint patterns through a Streamlit dashboard.

## Problem Statement

The project simulates a telecom brand intelligence use case for ZENDS Communications. Its objective is to reduce the manual effort involved in analysing customer feedback and to make relevant company and service information easier to retrieve when responding to customer concerns.

## Key Features

- **Sentiment classification:** predicts Positive, Negative, or Neutral sentiment.
- **Service category classification:** identifies the service area associated with a feedback message.
- **Policy and document retrieval:** searches indexed company documents using semantic embeddings and ChromaDB.
- **Context-grounded response generation:** uses retrieved context to answer customer questions.
- **Customer feedback interface:** accepts feedback and displays the analysis and response.
- **Brand analyst dashboard:** displays sentiment distribution, category-level complaint patterns, recommendations, and recent feedback monitoring.
- **Local model integration:** uses locally hosted Ollama models for embeddings and LLM responses.

## System Workflow

```text
Customer Feedback
       |
       v
Text Input in Streamlit
       |
       +------------------------------+
       |                              |
       v                              v
Sentiment Classification       Service Category Classification
(TF-IDF + Logistic Regression)  (TF-IDF + Logistic Regression)
       |                              |
       +---------------+--------------+
                       |
                       v
              Retrieve Relevant Context
                       |
             Query Embedding (Ollama)
                       |
                       v
                 ChromaDB Search
                       |
                       v
             Retrieved Document Chunks
                       |
                       v
             LLM Response Generation
                 (Ollama / Llama)
                       |
                       v
             Customer Response & Logging
                       |
                       v
             Brand Analyst Dashboard
```

## Technology Stack

| Area | Tools / Libraries |
|---|---|
| Programming language | Python |
| Data handling | Pandas, NumPy |
| Data visualisation | Matplotlib, Seaborn |
| NLP feature extraction | Scikit-learn TF-IDF |
| Classification | Scikit-learn Logistic Regression |
| Model persistence | Joblib |
| PDF text extraction | pypdf |
| Embeddings | Ollama `mxbai-embed-large` |
| Vector database | ChromaDB |
| LLM | Ollama `llama3.2:1b` |
| LLM API interface | OpenAI-compatible Python client configured for Ollama |
| Application | Streamlit |

## Machine Learning

Two separate text-classification pipelines are trained using the `feedback_text` field.

### 1. Sentiment Classification

Predicts one of the following sentiment labels:

- Positive
- Negative
- Neutral

### 2. Service Category Classification

Predicts one of the following six service categories:

- Mobile Network
- Broadband Service
- Billing & Payments
- Customer Support
- Mobile App Issues
- Service Activation

### Model pipeline

Both models use:

- `TfidfVectorizer`
- Unigrams and bigrams: `ngram_range=(1, 2)`
- Maximum vocabulary size: `max_features=5000`
- `LogisticRegression`
- `max_iter=1000`
- `class_weight='balanced'`
- An 80/20 train-test split with `random_state=42` and stratification

The sentiment and category models use separate TF-IDF vectorizers and separate Logistic Regression classifiers.

### Saved model artifacts

The application expects these four files in the `models/` directory:

- `tfidf_sent.pkl`
- `sentiment_model.pkl`
- `tfidf_cat.pkl`
- `category_model.pkl`

The notebook saves these artifacts using Joblib. Update the save paths in the notebook to match your local project directory before running it.

## Retrieval-Augmented Generation

The RAG component retrieves relevant information from two PDF documents:

- `company_profile.pdf`
- `telecom_knowledge_base.pdf`

### RAG pipeline

1. Extract text from the PDF files using `pypdf`.
2. Split the combined extracted text into chunks.
3. Generate embeddings for each chunk using Ollama's `mxbai-embed-large` model.
4. Store chunk text, IDs, and embeddings in a persistent ChromaDB collection named `mannual`.
5. Convert the user's question into an embedding.
6. Retrieve the top 3 relevant chunks from ChromaDB.
7. Add the retrieved context and question to the prompt.
8. Generate an answer using the Ollama `llama3.2:1b` model.

### RAG configuration recorded in the notebook

| Setting | Value |
|---|---|
| Source PDF documents | 2 |
| Extracted pages | 8 total |
| Combined extracted text | 14,449 characters |
| Chunk size | 500 characters |
| Chunk overlap | 50 characters |
| Number of chunks | 33 |
| Embedding model | `mxbai-embed-large` |
| Embedding dimensions | 1,024 |
| Vector database | ChromaDB |
| Collection name | `mannual` |
| Retrieved results | Top 3 |
| Generation model | `llama3.2:1b` |
| Generation temperature | 0.1 |

The RAG prompt instructs the model to answer using only the retrieved context and to state when the requested information is not present in the provided documents.

## Streamlit Application

The Streamlit application contains three navigation pages.

### 1. Home

- Shows system status for the classification models, Ollama, and ChromaDB.
- Displays feedback record and service category summary metrics.
- Describes the active ML and RAG pipeline.

### 2. Customer Feedback

- Accepts customer feedback through a text area.
- Predicts sentiment and service category.
- Displays the sentiment confidence when available.
- Retrieves relevant policy or service-document chunks.
- Produces a template-based or Ollama-assisted response.
- Supports logging submitted feedback.

The application also contains additional rule-based handling for mixed-sentiment messages, informational questions, and selected service-category keywords. These are application-level rules; they are not additional trained classification classes.

### 3. Brand Analyst Dashboard

- Displays sentiment and service-category analysis.
- Highlights negative feedback and recurring complaint patterns.
- Provides service-improvement recommendations based on observed feedback patterns and retrieved policy context.
- Includes recent feedback monitoring with periodic refresh.

## Dataset

The model notebook reads a CSV file named `new_data.csv`. The dataset contains three columns:

| Column | Description |
|---|---|
| `feedback_text` | Text of the customer feedback |
| `service_category` | Telecom service category associated with the feedback |
| `sentiment` | Positive, Negative, or Neutral label |

### Dataset snapshot from the notebook

- **Rows:** 5,175
- **Columns:** 3
- **Missing values:** 0
- **Duplicate rows:** 0
- **Unique feedback messages:** 5,175
- **Sentiment classes:** 3
- **Service categories:** 6

Sentiment distribution recorded in the notebook:

| Sentiment | Records |
|---|---:|
| Positive | 2,566 |
| Negative | 2,099 |
| Neutral | 510 |
| **Total** | **5,175** |

Service category distribution recorded in the notebook:

| Service category | Records |
|---|---:|
| Mobile Network | 884 |
| Broadband Service | 871 |
| Billing & Payments | 871 |
| Customer Support | 858 |
| Mobile App Issues | 853 |
| Service Activation | 838 |
| **Total** | **5,175** |

The dataset is synthetic. Performance on this dataset should not be interpreted as evidence of equivalent performance on real customer feedback.

## Model Evaluation

The notebook uses a stratified 80/20 train-test split and reports accuracy, classification reports, and confusion matrices. It also runs 5-fold stratified cross-validation.

### Test-set accuracy

| Model | Test accuracy |
|---|---:|
| Sentiment classifier | 99.71% |
| Service category classifier | 99.90% |

### 5-fold cross-validation

| Model | Metric | Mean validation score |
|---|---|---:|
| Sentiment | Accuracy | 99.40% |
| Sentiment | Macro F1-score | 99.35% |
| Sentiment | Weighted F1-score | 99.40% |
| Service category | Accuracy | 99.94% |
| Service category | Macro F1-score | 99.94% |
| Service category | Weighted F1-score | 99.94% |

These are the results recorded in the supplied notebook. Because the data is synthetic, further evaluation on independently collected, real-world feedback is necessary.

## Project Structure

The application code expects a structure similar to the following. Adjust folder and file names to match the files you include in your GitHub repository.

```text
telecom-ai-brand-intelligence/
│
├── app_local_c.py
│
├── notebooks/
│   ├── sentiment1.ipynb
│   └── rag.ipynb
│
├── models/
│   ├── tfidf_sent.pkl
│   ├── sentiment_model.pkl
│   ├── tfidf_cat.pkl
│   └── category_model.pkl
│
├── knowledge_base/
│   ├── company_profile.pdf
│   └── telecom_knowledge_base.pdf
│
├── data/
│   ├── new_data.csv
│   ├── zends_feedback_clean.csv
│   └── feedback_log.csv
│
├── vector_db/
│   └── chroma_db/
│
├── requirements.txt
└── README.md
```

**Important:** This is an expected/illustrative layout based on the notebook and application paths. Include only files that are actually present in your repository. The application code currently uses a hard-coded local ChromaDB path, so that path must be changed to a path relative to the project before moving the repository to another machine.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd telecom-ai-brand-intelligence
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn pypdf chromadb openai
```

## Running the Project

### 1. Install and start Ollama

Install Ollama from its official website and make sure the local Ollama service is running.

Pull the models used by the project:

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2:1b
```

The application and notebook connect to the local Ollama OpenAI-compatible endpoint:

```text
http://localhost:11434/v1
```

### 2. Prepare the model files

Run the sentiment-model notebook after updating the dataset path and model output paths for your machine. Confirm that the four `.pkl` files are present in `models/`.

### 3. Prepare the vector database

Run the RAG notebook after updating the PDF input folder and ChromaDB persistence path. Confirm that the `mannual` collection contains the document chunks.

### 4. Launch Streamlit

```bash
streamlit run app_local_c.py
```

Open the local URL printed by Streamlit in your browser.

## Configuration and Notes

Before running the project on another computer:

- Update the CSV path used in `sentiment1.ipynb`.
- Update the model output paths in `sentiment1.ipynb`.
- Update the PDF folder and ChromaDB persistence path in `rag.ipynb`.
- Update `CHROMA_DIR` in `app_local_c.py`; it currently points to a machine-specific Windows directory.
- Make sure the model files, ChromaDB data, and required CSV/PDF files are available at the configured locations.
- Keep Ollama running locally and ensure both required models have been pulled.
- Do not commit private data, secrets, or unnecessary large generated files. Consider using Git LFS or release assets for large model/database artifacts, or provide instructions for generating them locally.

## Limitations and Future Improvements

### Current limitations

- The training dataset is synthetic; model performance may not generalise to real customer language.
- The RAG notebook stores generic PDF source metadata for chunks. More detailed source and page metadata would improve traceability of retrieved answers.
- Ollama and the vector database are configured for local execution; the application is not configured as a hosted cloud service in the supplied code.
- Mixed-sentiment handling and selected category corrections in the app use rules in addition to the trained classifiers.

### Possible future improvements

- Evaluate and retrain the models using real, representative telecom feedback.
- Add more robust text cleaning, spelling variation handling, and multilingual support.
- Preserve document filename and page number metadata for every RAG chunk.
- Evaluate retrieval quality and generated-answer faithfulness with a defined test set.
- Make all file and database paths configurable through environment variables or a configuration file.
- Add authentication, access controls, and production-grade feedback storage before real customer deployment.

## Acknowledgements

This project was developed as a simulated telecom brand intelligence use case using machine learning, NLP, retrieval-augmented generation, and Streamlit.
