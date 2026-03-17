# 🏦 FinGuard AI: Advanced Loan Application Assessment System

**FinGuard AI** is a professional-grade credit appraisal system that replicates the bank lending process. It combines **XGBoost Machine Learning** with a **Rule-Based Expert System** to analyze business financial health, GST compliance, and balance sheet ratios.

Developed for the B.Tech 2026 Project by **Arjun Sarkar** & **Urvashi Bohare**.

---

## 🚀 Key Features

* **Hybrid Decision Engine:** Uses a combination of AI (XGBoost) and Hard-Compliance Rules.
* **Balance Sheet Analysis:** Automated calculation of Current Ratio, Debt-to-Equity, and ROA.
* **GST Discipline Scoring:** Evaluates borrower authenticity through tax filing history.
* **Risk Dashboard:** Interactive Streamlit interface with real-time risk gauges.
* **Modern UI:** Dark-mode "Red Edition" aesthetic for a premium banking experience.

---

## 🛠️ Technology Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Frontend** | HTML5 / CSS3 (Landing Page) |
| **Interface** | Streamlit |
| **ML Model** | XGBoost Classifier |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |

---

## 📂 Project Structure

```text
Loan_AI_Project/
├── data/
│   └── final_hybrid_loan_data.csv  # Synthetic dataset for training
├── app.html                        # Red-themed Landing Page
├── app.py                          # Streamlit Dashboard (Main App)
├── data_gen.py                     # Script to generate hybrid data
├── train_model.py                  # AI training script (generates .pkl)
├── loan_model.pkl                  # The trained AI brain
├── label_encoder.pkl               # Target label translator
└── requirements.txt                # List of dependencies