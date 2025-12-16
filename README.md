# mental-health-prediction
This project applies machine learning techniques to real-world social survey data, based on variables commonly found in the Understanding Society study, to predict poor mental health risk.
## 🧠 Dataset Description
**Understanding Society** is a longitudinal household survey that collects annual data on:
- Mental and physical health
- Socio-economic status
- Employment and income
- Lifestyle and behavioral factors
- Demographic characteristics

In this project:
- A **single wave (one-year snapshot)** of the dataset was used  
- The goal was to perform **cross-sectional analysis**, not longitudinal tracking
- The target variable represents **binary mental health status** (Good vs Poor)

> ⚠️ Note: The original dataset is not included in this repository due to size and data usage restrictions.

---

## 🔧 Methodology
The following steps were applied:
- Data preprocessing and feature engineering
- Proper train/test split to avoid data leakage
- Encoding strategies (binary encoding, mean target encoding, standardization)
- Class imbalance handling using SMOTE
- Model training and tuning

### Models evaluated:
- Logistic Regression
- Random Forest
- LightGBM (best-performing model)

Model performance was evaluated using:
- ROC AUC
- Gini coefficient
- Precision, Recall, F1-score
- Confusion Matrix and ROC curves

---

## 📈 Results
The LightGBM model achieved the highest performance in terms of ROC AUC and Gini coefficient, demonstrating strong predictive capability for identifying individuals at higher risk of poor mental health.

Key visual outputs included:
- ROC curve comparison across models
- Feature importance analysis
- Confusion matrix of the best model
