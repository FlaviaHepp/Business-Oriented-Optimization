# 💰Profit-Driven Customer Churn Prediction for Banking

## 📌Business Context

Customer churn is one of the most critical challenges in the banking industry.  
Acquiring new customers is significantly more expensive than retaining existing ones.

👉 However, **not all customers have the same value**.

This project focuses on a key business question:

> **Which customers should we prioritize to maximize retention profit?**

---

## 🎯Project Objective

Build a machine learning model that:

- Predicts customer churn
- Identifies high-risk customers
- Optimizes retention strategy based on **profitability**
- Supports **data-driven decision making**

---

## 🧠Approach

### 🔹 Data Preparation
- Data cleaning and normalization
- Removal of irrelevant features (`customerid`, `surname`, etc.)
- One-hot encoding of categorical variables

### 🔹 Exploratory Data Analysis (EDA)
- Churn behavior by age, balance, and number of products
- Identification of patterns linked to customer exit

### 🔹 Modeling
- Algorithm: **Random Forest Classifier**
- Handling class imbalance with `class_weight="balanced"`
- Stratified train/test split

### 🔹 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

---

## 💰Business-Oriented Optimization

Traditional models use a default threshold (0.5).  
This project introduces a **profit-based decision strategy**.

### ✔ Key Idea:
Instead of predicting churn →  
👉 **Optimize actions to maximize profit**

---

## 💸Profit Function

The model incorporates business constraints:

- 💵 Gain per retained customer: **$100**
- 📞 Cost per contact: **$20**

``python
Profit = (True Positives * Gain) - (Contacts * Cost)

📊 Threshold Optimization
- Multiple probability thresholds evaluated
- Profit calculated for each scenario
- Best threshold selected dynamically

👉 Result:
- Maximized ROI
- Reduced unnecessary contact costs
- 
🤖 Model vs Random Strategy

The model is compared against a baseline:

🤖 ML-driven targeting
🎲 Random selection
✅ Outcome:

The model significantly improves:
- Profitability
- Targeting efficiency
- Resource allocation

🔍 Model Interpretability (SHAP)

To ensure transparency and business usability, the model is interpreted using SHAP (SHapley Additive Explanations).

🔹 What it provides:
Global feature importance
Individual prediction explanations
Identification of churn drivers

## 💡Business Impact:
Enables personalized retention strategies
Bridges the gap between ML and decision-making

## 📈Visualizations
Customer behavior analysis (Age, Balance, Products)
Confusion Matrix
Profit vs Threshold curve
SHAP feature importance plots

## 🛠Tech Stack
Python
Pandas / NumPy
Scikit-learn
Matplotlib / Seaborn
SHAP

## 📂Project Structure
│── comportamiento_clientes_bancarios.py
│── Dataset.csv
│── README.md
│── requirements.txt

## ▶️How to Run
pip install -r requirements.txt
python comportamiento_clientes_bancarios.py

## 🚀Key Insights
Churn prediction alone is not sufficient
Profit-driven strategies outperform probability-based decisions
Threshold tuning has a direct impact on ROI
Model interpretability is essential for real-world adoption

## 🧠Business Takeaways

This project demonstrates that:

✔ Data Science must align with business objectives
✔ Decisions should be driven by economic impact, not just accuracy
✔ Interpretable models enable actionable strategies

## 📬About Me

Data Scientist focused on:

💰 Revenue Optimization
📊 Business Analytics
🧠 Customer Intelligence

I build data solutions that drive measurable business impact.
