# BBC News Classification (TF-IDF + Machine Learning)

A machine learning project that classifies **BBC news articles** into categories using **TF-IDF vectorization** and a supervised **scikit-learn** model.  
Includes a simple **Gradio** app for interactive predictions.

---

## Features

- Train a text classification model from CSV datasets
- TF-IDF feature extraction for text
- Label encoding for categories
- Save and reuse trained artifacts (`.pkl`)
- Run an interactive demo with Gradio

---

## Project Structure

DATH-251-BBC/
├─ data-bbc/
│ ├─ train.csv
│ ├─ val.csv
│ └─ test.csv
├─ .gradio/ # Gradio cache/output (usually ignored)
├─ app.py # Gradio UI for prediction
├─ bbc_news_tfidf_ml.py # Training / evaluation pipeline
├─ tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├─ text_model.pkl # Saved trained model
├─ label_encoder.pkl # Saved label encoder
├─ classification_results.html # Evaluation report (optional)
└─ README.md


---

## Dataset

The dataset is stored in `data-bbc/`:

- `train.csv` — training set
- `val.csv` — validation set
- `test.csv` — test set

> Note: Column names may vary depending on the dataset version.  
> If needed, update the column names inside `bbc_news_tfidf_ml.py`.

---

## Requirements

- Python 3.9+ (recommended)
- pip
Install dependencies:

```bash
pip install pandas scikit-learn gradio
```
---
## How to run
1. Clone the repository

git clone https://github.com/KYTO1111501/dath-251-bbc-news-classification.git
cd dath-251-bbc-news-classification


2. Create virtual environment (optional but recommended)

python -m venv .venv


3. Activate virtual environment

Windows:
.venv\Scripts\activate

Mac / Linux:
source .venv/bin/activate


4. Install dependencies

pip install pandas scikit-learn gradio


5. Train the model (optional)

python bbc_news_tfidf_ml.py


6. Run the Gradio application

python app.py
