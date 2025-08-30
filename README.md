# 🏥 Diabetes Hospital Readmission Analysis

## 📌 Project Overview
This project analyzes hospital readmission patterns using the **Diabetes 130-US hospitals dataset (1999–2008)**.  
The goal is to identify demographic factors associated with 30-day readmissions and build a predictive model using logistic regression. The workflow demonstrates data cleaning, exploratory analysis, feature engineering, modeling, evaluation, and clinical interpretation.

---

## 🔬 Motivation
Hospital readmissions are costly and affect patient outcomes. Understanding which demographic groups are at higher risk can help healthcare providers implement targeted interventions and improve care quality.

---

## 📂 Dataset
- **Source**: [UCI Diabetes 130-US hospitals dataset / Kaggle link]  
- **Size**: ~100,000 patient encounters (subset can be used for demonstration)  
- **Features used**:  
  - Demographics: age, race, gender  
  - Clinical info: time in hospital, number of diagnoses  
  - Target: `readmitted` (converted to binary: readmitted vs. not readmitted)  
- **Ethics**: Data is de-identified; no PHI is used.

---

## 🛠️ Methodology

1. **Data Cleaning & Preprocessing**  
   - Removed missing/unknown values  
   - Encoded categorical variables (age midpoints, gender numeric, one-hot race)  
   - Created binary target variable for readmission  

2. **Exploratory Data Analysis (EDA)**  
   - Readmission distributions overall and by demographics  
   - Correlation heatmaps  
   - Visualized key demographic patterns  

3. **Modeling**  
   - Logistic regression with scaled features  
   - Stratified train-test split (70/30)  
   - Evaluated using ROC-AUC, classification report, and confusion matrix  

4. **Feature Importance & Clinical Interpretation**  
   - Coefficients and odds ratios calculated  
   - Key risk factors identified and visualized  
   - Provided actionable clinical recommendations  

---

## 📈 Results
- **Model Performance**:  
  - ROC-AUC: `XX`  
  - Accuracy, precision, recall, and F1-score reported in `classification_report`  
- **Key Risk Factors**: Top demographic and clinical features increasing readmission risk  
- **Visualizations**:  
  - Readmission distributions by gender, race, and age  
  - Correlation heatmap  
  - Confusion matrix & ROC curve  
  - Top feature coefficients  

*(Insert sample plots/images here using GitHub Markdown: `![Title](results/plot.png)`) *

---

## 💡 Clinical Insights
- Certain demographic groups (e.g., older age, specific racial groups) have higher readmission odds  
- Longer hospital stays and higher number of diagnoses are associated with increased risk  
- Recommendations: enhanced discharge planning, targeted follow-up, culturally appropriate care pathways  

**Limitations:**  
- Demographics alone do not capture full clinical complexity  
- Temporal trends (1999–2008) may differ from current healthcare landscape  
- Model explains limited variance; additional features like comorbidities or medications could improve predictions  

---

## ⚙️ How to Run
1. Clone repository:  
```bash
git clone https://github.com/<your-username>/hospital-readmission-analysis.git
cd hospital-readmission-analysis
