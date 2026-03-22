# 💳 Credit Card Approval Prediction: Machine Learning Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Sklearn](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-Random_Forest-green)
![Technique](https://img.shields.io/badge/Technique-SMOTE_Balancing-purple)

## 📖 Project Overview

Financial institutions need robust methods to assess credit risk. This project utilizes a **Random Forest Classifier** to predict whether a credit card applicant is a "Good" or "Bad" client.

The core challenge of this dataset was **Class Imbalance**. The vast majority of applicants are "good" payers, making it difficult for standard models to detect the "bad" payers (defaulters). This project overcomes that using **SMOTE (Synthetic Minority Over-sampling Technique)**.

---

## 🧠 Methodology & Definitions

### 1. Defining the Target Variable ("Good" vs "Bad")
The raw dataset provided payment history (0-29 days late, 30-59 days, >60 days, etc.).
*   **Bad Client (Target = 1):** Any customer who has been overdue by **60+ days** (Status 2, 3, 4, or 5) at least once in their history.
*   **Good Client (Target = 0):** Customers who have never exceeded a 60-day delinquency.

### 2. Handling Imbalanced Data
Initial analysis showed a severe imbalance (approx. 98% Good / 2% Bad).
*   **The Problem:** A model trained on this would achieve 98% accuracy simply by guessing "Good" every time, failing to detect actual risk.
*   **The Solution:** I applied **SMOTE**. This algorithm generates synthetic examples of "Bad" customers based on the vector neighbors of existing bad customers, balancing the training set to a 50/50 ratio before training.

---

## ⚙️ The Pipeline

1.  **Data Cleaning:** Handling missing values and removing irrelevant columns (e.g., Phone ID).
2.  **Feature Engineering:** Encoding categorical variables (Gender, Income Type, Education) into numeric formats using Label Encoding.
3.  **Resampling:** applying SMOTE to the training data.
4.  **Modeling:** Training a **Random Forest Classifier**.
5.  **Evaluation:** assessed using **Recall** and **F1-Score** rather than simple Accuracy.

---

## 📊 Key Findings

The model identified the following as the top predictors for creditworthiness:
1.  **Income Total:** Higher income correlates with lower risk.
2.  **Age:** Older applicants tend to be more stable payers.
3.  **Employment Length:** Years employed is a strong indicator of stability.

## 🚀 How to Run
1.  Clone the repository.
2.  Download the `application_record.csv` and `credit_record.csv` files.
3.  Run `Credit_Model.ipynb`.

---
**Author:** Neelam
