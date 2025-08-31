# 🏥📊 Diabetes Hospital Readmission Analysis  

## 📋 Project Overview  
This project analyzes **hospital readmission patterns in diabetic patients** using data from **130 US hospitals (1999–2008)**.  
The goal is to identify **demographic and clinical risk factors** that predict readmission within 30 days, enabling healthcare providers to design targeted interventions and improve patient outcomes.  

**Key Research Question:**  
👉 Which demographic factors best predict hospital readmission risk in diabetic patients?  

---

## 🎯 Objectives  
- Explore demographic patterns in hospital readmissions  
- Build predictive models to identify high-risk patients  
- Provide actionable insights for clinical decision-making  
- Support evidence-based discharge planning protocols  

---

## 📊 Dataset Information  
- **Source**: [UCI Diabetes 130-US hospitals dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)  
- **Population**: Diabetic patients with hospital encounters  
- **Key Variables**: race, gender, age, length of stay, number of diagnoses, readmission status  
- **Target Variable**: Binary readmission flag (`readmitted` vs `not readmitted`)  

| Variable          | Type        | Description                                      |
|-------------------|------------|--------------------------------------------------|
| race              | Categorical | Patient race/ethnicity                          |
| gender            | Categorical | Patient gender (Male/Female)                    |
| age               | Categorical | Age group in 10-year brackets                   |
| time_in_hospital  | Numeric     | Length of stay (days)                           |
| number_diagnoses  | Numeric     | Total number of diagnoses                       |
| readmitted        | Categorical | Readmission status (NO, <30, >30)               |

---

## 🛠️ Technical Stack  
- **Python 3.7+**  
- **pandas, numpy** → data manipulation & numerical computing  
- **matplotlib, seaborn** → data visualization  
- **scikit-learn** → machine learning models & evaluation  
- **Jupyter Notebook** → interactive workflow  

---

## 📈 Analysis Pipeline  

### 1. Data Preprocessing  
- Handled missing values and unknown entries  
- Encoded categorical variables (age, gender, race)  
- Created binary target variable for readmission  

### 2. Exploratory Data Analysis (EDA)  
- Readmission distribution overall and by demographics  
- Correlation analysis across key variables  
- Descriptive statistics and demographic breakdowns  

### 3. Predictive Modeling  
- **Algorithm**: Logistic Regression  
- **Features**: Demographic + clinical indicators  
- **Evaluation**: Classification report, ROC-AUC, confusion matrix  
- **Interpretation**: Odds ratios, feature coefficients  

### 4. Clinical Insights  
- Ranked features by importance  
- Identified major risk factors  
- Proposed targeted clinical recommendations  
- Discussed model limitations and next steps  

---

## 📊 Results & Insights  

### Readmission Outcomes  
- Three outcomes observed: no readmission, <30-day readmission, >30-day readmission  
- Early (≤30 days) readmissions were **significantly higher** → signals **acute post-discharge complications** (e.g., infections, cardiac failure, medication issues)
- <img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/88a3e223-c905-4241-9d4d-59c631124229" />


### Demographic Patterns  
- **Gender**: Readmission rates proportional to dataset distribution → broad interventions preferred  
- **Age**: Highest readmission in **50–70 year-old patients**, requiring targeted care planning
- <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/00c09bca-f632-4895-adf1-4f2af2bc4b60" />

-<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/ea472928-1b38-4fa6-aa8a-f897431fcd9e" />

### Correlation Analysis  
- **Time in hospital** & **number of diagnoses** had the **strongest positive correlation** with readmission  
- **Age** showed moderate positive correlation  
- *Heatmap coding*: White → 0, Blue → positive, Red → negative

- <img width="495" height="427" alt="image" src="https://github.com/user-attachments/assets/3fe452f8-1081-44af-951a-76ebf7f9bc64" />


### Model Performance  
- Logistic regression ROC-AUC = **0.564** → weak predictive ability, slightly above random guessing  
- Confusion matrix revealed the highest risk in **false negatives** (missed readmissions)

- <img width="597" height="488" alt="image" src="https://github.com/user-attachments/assets/0ade0bbd-551a-4a09-a572-dc9b0a9c163b" />

-<img width="376" height="300" alt="image" src="https://github.com/user-attachments/assets/a2481e9d-4acd-45a6-b325-752636039e8a" />



### Feature Importance  
- Strongest predictors:  
  - **Number of diagnoses** ↑ risk  
  - **Time in hospital** ↑ risk  
- Coefficients interpreted as odds ratios to quantify effect size

- <img width="752" height="448" alt="image" src="https://github.com/user-attachments/assets/3cbf9223-42dd-4a47-ad3a-f2e07cafb8c7" />
  

---

## 🔮 Next Steps  
- Add **comorbidity, medication, and lab data** for richer features  
- Try **advanced models** (Random Forest, Gradient Boosting, Neural Networks)  
- Address **class imbalance** (SMOTE, resampling)  
- Deploy insights via a **Streamlit/Dash interactive dashboard**  

---

## 💡 Key Takeaway  
Even with modest performance (AUC = 0.564), this project demonstrates the **end-to-end AI/ML pipeline in healthcare**:  
- Data cleaning & preprocessing  
- Exploratory analysis  
- Predictive modeling  
- Evaluation & clinical interpretation  

This highlights how **AI can inform clinical decision-making** while emphasizing the need for **feature-rich data and model refinement** in sensitive healthcare contexts.  
