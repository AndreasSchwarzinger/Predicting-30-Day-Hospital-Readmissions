Diabetes Hospital Readmission Analysis 🏥📊
A comprehensive clinical analytics project examining demographic predictors of hospital readmission among diabetic patients using machine learning and statistical analysis.
📋 Project Overview
This project analyzes hospital readmission patterns in diabetic patients using data from 130 US hospitals (1999-2008). The goal is to identify demographic risk factors that predict readmission within 30 days, enabling healthcare providers to implement targeted interventions and improve patient outcomes.
Key Research Question
Which demographic factors best predict hospital readmission risk in diabetic patients?
🎯 Objectives

Explore demographic patterns in hospital readmissions
Build predictive models to identify high-risk patients
Provide actionable insights for clinical decision-making
Support evidence-based discharge planning protocols

📊 Dataset Information

Source: Diabetes 130-US hospitals dataset (1999-2008)
Population: Diabetic patients with hospital encounters
Key Variables: Race, gender, age, length of stay, number of diagnoses, readmission status
Target Variable: Binary readmission flag (readmitted vs not readmitted)

Dataset Features
VariableTypeDescriptionraceCategoricalPatient race/ethnicitygenderCategoricalPatient gender (Male/Female)ageCategoricalAge group in 10-year bracketstime_in_hospitalNumericLength of stay (days)number_diagnosesNumericTotal number of diagnosesreadmittedCategoricalReadmission status (NO, <30, >30)
🛠️ Technical Stack

Python 3.7+
Pandas - Data manipulation and analysis
NumPy - Numerical computing
Matplotlib & Seaborn - Data visualization
Scikit-learn - Machine learning models and evaluation
Jupyter Notebook - Interactive development environment

📈 Analysis Pipeline
1. Data Preprocessing

Handle missing values and unknown entries
Encode categorical variables (gender, age groups)
One-hot encode race categories
Create binary readmission target variable

2. Exploratory Data Analysis

Readmission Distribution: Overall readmission patterns
Demographic Breakdowns: Readmission rates by gender, age, race
Correlation Analysis: Relationships between variables
Statistical Summaries: Descriptive statistics by demographic groups

3. Predictive Modeling

Algorithm: Logistic Regression
Features: Demographic variables + clinical indicators
Evaluation: Classification metrics, ROC-AUC, confusion matrix
Interpretation: Odds ratios and confidence intervals

4. Clinical Insights

Feature importance ranking
Risk factor identification
Clinical recommendations
Model limitations assessment

🔍 Key Visualizations
Graph Outputs Explained:

Readmission Distribution (Pie Chart)

Shows overall readmission rates in the population
Helps understand baseline risk


Gender-Based Readmission (Bar Chart)

Compares readmission rates between males and females
Identifies gender-specific risk patterns


Age-Stratified Readmission (Stacked Bars)

Shows readmission patterns across age groups
Reveals age-related risk escalation


Correlation Matrix (Heatmap)

Numbers range from -1 to +1
Values closer to ±1 indicate stronger relationships
Blue = positive correlation, Red = negative correlation


Confusion Matrix (Heatmap)

Shows model prediction accuracy
Critical for understanding missed high-risk patients


ROC Curve

AUC score indicates overall model performance
Values: 0.5 (random) to 1.0 (perfect)


Feature Importance (Bar Chart)

Shows which demographic factors most strongly predict readmission
Positive coefficients = increased risk, Negative = decreased risk



🚀 Getting Started
Prerequisites
bashpip install pandas numpy matplotlib seaborn scikit-learn
Usage

Clone the repository
bashgit clone https://github.com/yourusername/diabetes-readmission-analysis.git
cd diabetes-readmission-analysis

Download the dataset

Place diabetic_data.csv in the project root directory
Dataset available from UCI Machine Learning Repository


Run the analysis
bashpython diabetes_readmission_analysis.py


Expected Outputs

Multiple visualization plots
enhanced_demographics_readmission_results.csv - Feature coefficients and odds ratios
model_performance_metrics.csv - Model evaluation metrics
Console output with detailed clinical insights

📊 Sample Results
Model Performance Metrics

AUC-ROC: 0.XXX (model discrimination ability)
Sensitivity: XX% (percentage of readmissions correctly identified)
Specificity: XX% (percentage of non-readmissions correctly identified)

Key Risk Factors (Example)
Demographic FactorOdds RatioClinical InterpretationAge (per decade)1.XXXX% increased readmission oddsMale Gender1.XXXX% higher risk vs femalesRace Category X1.XXXX% higher risk vs reference group
🎯 Clinical Applications
Immediate Use Cases

Risk Stratification: Identify high-risk patients at discharge
Resource Allocation: Target interventions to vulnerable populations
Quality Improvement: Reduce preventable readmissions
Care Coordination: Enhanced follow-up for high-risk demographics

Implementation Recommendations

Integrate demographic risk scores into discharge planning
Develop culturally appropriate care pathways
Implement targeted follow-up protocols
Monitor outcomes by demographic subgroups

⚠️ Limitations & Considerations
Model Limitations

Demographics alone capture limited variance in readmission risk
Historical data (1999-2008) may not reflect current healthcare landscape
Missing important clinical variables (comorbidities, medications, social determinants)
Potential bias in historical healthcare data

Clinical Considerations

Model should supplement, not replace, clinical judgment
Requires validation in current healthcare settings
Consider ethical implications of demographic-based risk stratification
Regular model retraining needed as healthcare evolves
