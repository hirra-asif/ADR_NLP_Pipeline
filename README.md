# Adverse Drug Reaction NLP pipeline 💊

This repository contains two Jupyter notebooks that implement **Adverse Drug Reaction (ADR) detection** on social media text data using both traditional machine learning techniques and deep learning models, with integrated **SHAP** and **LIME** explainability tools. The project classifies whether social media posts indicate the presence of an ADR.

**Key result:** fine-tuned RoBERTa reached **0.90 balanced accuracy and 0.97 ROC AUC** on the combined dataset (CADEC + PsyTAR + Reddit), compared with **0.83 / 0.91** for the best classical baseline.

---

## Notebooks

### 1) `Machine Learning` 📈
Implements ADR detection using traditional Machine Learning techniques including **Logistic Regression**, **Naive Bayes**, **Random Forest**, and **XGBoost**.

**Includes:**
- Text preprocessing (tokenization, stopword removal, stemming/lemmatization)
- Feature extraction using **TF-IDF**
- **Chi-square** feature selection
- **SMOTE** for handling class imbalance
- **5-fold GridSearchCV** for hyperparameter tuning
- Model evaluation & plots (accuracy, precision, recall, F1-score)
- Model explainability with **SHAP** to interpret feature contributions

---

### 2) `Deep Learning` 🧠
Implements ADR detection using **Transformer-based Deep Learning models** (e.g., **DeBERTa**, **RoBERTa**).

**Includes:**
- Data preprocessing for social media text (drug name & dosage masking, cleaning, tokenization, sequence padding)
- Fine-tuning **RoBERTa / DeBERTa** with Hugging Face Transformers
- Model training with validation and performance metrics (accuracy, precision, recall, F1-score)
- Saving fine-tuned models, tokenizers, and evaluation results
- Model explainability with **SHAP** and **LIME** to visualise token-level importance and identify key ADR indicators

---

## Results 📊

Held-out test set (80/20 split), threshold 0.5. Balanced accuracy is reported because the datasets are imbalanced (for example, CADEC is ~94% ADR-positive).

| Dataset | RoBERTa (BA / AUC) | DeBERTa-v3-small (BA / AUC) |
| --- | :---: | :---: |
| CADEC | 0.86 / 0.97 | 0.81 / 0.97 |
| PsyTAR | 0.88 / 0.95 | 0.88 / 0.96 |
| Reddit | 0.80 / 0.89 | 0.79 / 0.89 |
| **Combined** | **0.90 / 0.97** | **0.90 / 0.96** |

Best classical baseline on the combined dataset (TF-IDF + XGBoost, tuned): balanced accuracy 0.83, ROC AUC 0.91.

---

## How to run ▶️

The notebooks were developed in **Google Colab** (GPU runtime for the deep learning notebook).

1. Download the datasets (links below) and save them to your Google Drive.
2. Open a notebook in Colab, mount your Drive, and update the file paths in the first cell.
3. Run all cells. Fine-tuned models and metrics are saved back to your Drive.

Main dependencies: `pandas`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `transformers`, `torch`, `shap`, `lime`.

---

## Datasets Used 📄

- **CADEC v2** — *CSIRO Annotated Corpus for Adverse Drug Event Detection*: [https://data.csiro.au/collection/csiro:62387](https://data.csiro.au/collection/csiro:62387)  
- **PsyTAR** — *Adverse Drug Reaction Corpus for Psychiatric Medications*: [https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp](https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp)  
- **Reddit ADR Data** — Medical and drug related subreddit posts collected via the **Reddit API** and self-labeled for ADR presence

No dataset files are included in this repository. CADEC and PsyTAR are available from the links above under their own terms; the Reddit data is not redistributed.

---

## Figures

**Model comparison across datasets**

<img width="1790" height="594" alt="Model comparison across datasets" src="https://github.com/user-attachments/assets/003d5558-195b-4673-b5d0-b3cfa605e83d" />

---

**Evaluation metrics**

<img width="1010" height="549" alt="Evaluation metrics" src="https://github.com/user-attachments/assets/29c7ceb4-7348-4b3e-a633-a1d42555ff03" />

---

**Results summary**

<img width="1360" height="239" alt="Results summary" src="https://github.com/user-attachments/assets/b37e02bd-c2c9-4a02-b07c-5d1ad4666570" />
