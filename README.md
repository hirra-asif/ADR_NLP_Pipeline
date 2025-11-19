# Adverse Drug Reaction NLP pipeline 💊

This repository contains two Jupyter notebooks that implement **Adverse Drug Reaction (ADR) detection** on social media text data using both traditional machine learning techniques and deep learning models, with integrated **SHAP** and **LIME** explainability tools. The project classifies whether social media posts indicate the presence of an ADR.


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

## Datasets Used 📄

- **CADEC v2** — *CSIRO Annotated Corpus for Adverse Drug Event Detection*: [https://data.csiro.au/collection/csiro:62387](https://data.csiro.au/collection/csiro:62387)  
- **PsyTAR** — *Adverse Drug Reaction Corpus for Psychiatric Medications*: [https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp](https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp)  
- **Reddit ADR Data** — Medical and drug related subreddit posts collected via the **Reddit API** and self-labeled for ADR presence

---
<img width="1790" height="594" alt="metrics2" src="https://github.com/user-attachments/assets/003d5558-195b-4673-b5d0-b3cfa605e83d" />

---

<img width="1010" height="549" alt="metrics" src="https://github.com/user-attachments/assets/29c7ceb4-7348-4b3e-a633-a1d42555ff03" />

---

<img width="1360" height="239" alt="Screenshot 2025-08-18 at 14 55 59" src="https://github.com/user-attachments/assets/b37e02bd-c2c9-4a02-b07c-5d1ad4666570" />
