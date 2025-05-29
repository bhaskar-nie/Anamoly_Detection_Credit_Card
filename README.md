# Anomaly Detection in Credit Card Transactions

**Author:** Bhaskar Anand  
**Project Type:** Machine Learning – Unsupervised Learning  
**Domain:** Fraud Detection | Financial Data  
**Dataset:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 📌 Overview

This project implements an **Anomaly Detection System** to flag potentially fraudulent credit card transactions. The system is trained only on **authentic transactions** and is designed to detect **rare, anomalous behavior** using a **Multivariate Gaussian Model**. Given the highly imbalanced nature of the dataset, we prioritize **recall** and use the **F2-score** as the evaluation metric to minimize false negatives.

---

## 🧠 Problem Statement

Detect credit card frauds in a dataset where:
- Only **0.173%** of transactions are fraudulent.
- Fraudulent behavior is **rare**, **evolving**, and **diverse**.
- Traditional supervised models struggle due to the **class imbalance**.

---

## ⚙️ Methodology

### 1. **Data Understanding**
- Transactions from European cardholders over two days.
- 31 features: `Time`, `Amount`, `Class`, and `V1`–`V28` (PCA-transformed).

### 2. **Feature Engineering**
- Extracted `Hour` from `Time`.
- Applied `log10` transformation on `Amount` to reduce skewness.

### 3. **Feature Selection**
- Retained only features with clearly different distributions across fraud and non-fraud cases:
  - `V4`, `V11`, `V12`, `V14`, `V16`, `V17`, `V18`, `V19`, `Hour`

### 4. **Modeling Approach**
- Fit a **Multivariate Normal Distribution** on authentic transactions.
- For a new transaction, compute its probability density.
- If this density is **below a chosen threshold**, flag it as **fraudulent**.

### 5. **Threshold Optimization**
- Threshold `ε` tuned using the **F2-score** on a validation set.
- Optimal threshold ≈ `0.009^9 ≈ 3.87 × 10⁻¹⁹`

---

## 📊 Evaluation

| Metric | Validation Set | Test Set |
|--------|----------------|----------|
| F2-score | **0.8347** | **0.8165** |

- **Confusion Matrices** and metric plots included in notebook.
- High recall ensures fewer undetected frauds.

---

## 🧪 Data Splitting

- **Training Set**: 80% of authentic transactions only  
- **Validation Set**: 10% authentic + 50% fraudulent  
- **Test Set**: 10% authentic + 50% fraudulent

---

## 📁 Project Structure

```
📦 credit-card-fraud-anomaly-detection
│
├── 📜 README.md
├── 📓 anomaly_detection_notebook.ipynb
├── 📁 images/
│   ├── confusion_matrix_validation.png
│   ├── confusion_matrix_test.png
│   └── threshold_tuning_plot.png
└── 📄 creditcard.csv (via Kaggle)
```

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Multivariate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
- [F2 Score](https://en.wikipedia.org/wiki/F-score)
- [Thudumu et al., 2020 – Dimensionality & Anomaly Detection](https://doi.org/10.1186/s40537-020-00320-x)

---

## ✅ Key Takeaways

- **Anomaly detection** is ideal for rare-event problems like fraud detection.
- **Feature transformation** and **selection** are crucial before modeling.
- **F2-score** is more appropriate than accuracy for fraud detection tasks.
- The approach generalizes well, making it suitable for deployment pipelines.

---

## 🚀 Future Enhancements

- Explore **autoencoders** or **isolation forests** for comparison.
- Implement **real-time streaming detection**.
- Build a web dashboard using Flask/Streamlit for visual insights.

---
