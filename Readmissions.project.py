# Clinical Analytics Mini-Project
# Dataset: Diabetes 130-US hospitals (1999-2008)
# Goal: Explore demographic predictors of hospital readmission

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =========================
# 1. Load dataset
# =========================
try:
    df = pd.read_csv("diabetic_data.csv")
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print("Dataset not found. Please ensure 'diabetic_data.csv' is in the working directory.")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 5000
    df = pd.DataFrame({
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.choice(['[50-60)', '[60-70)', '[70-80)', '[80-90)'], n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'number_diagnoses': np.random.randint(1, 16, n_samples),
        'readmitted': np.random.choice(['NO', '<30', '>30'], n_samples, p=[0.6, 0.25, 0.15])
    })
    print("Using simulated data for demonstration")

print("\nFirst few rows:")
print(df.head())

# =========================
# 2. Data Quality Assessment
# =========================
print("\n" + "=" * 50)
print("DATA QUALITY ASSESSMENT")
print("=" * 50)

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
    if df[col].nunique() < 10:
        print(f"  Values: {df[col].unique()}")

# =========================
# 3. Enhanced Data Preprocessing
# =========================
print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# Focus on demographics + readmission
cols = ["race", "gender", "age", "time_in_hospital", "number_diagnoses", "readmitted"]
df_clean = df[cols].copy()

# Handle missing/unknown values
print(f"Records before cleaning: {len(df_clean)}")
df_clean = df_clean[df_clean["race"] != "?"]
df_clean = df_clean[df_clean["gender"].isin(["Male", "Female"])]
print(f"Records after cleaning: {len(df_clean)}")

# Create binary readmission target
df_clean["readmission_flag"] = df_clean["readmitted"].apply(lambda x: 1 if x != "NO" else 0)

# Enhanced age encoding with midpoint values
age_mapping = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95
}
df_clean["age_numeric"] = df_clean["age"].map(age_mapping)

# Gender encoding
df_clean["gender_numeric"] = df_clean["gender"].map({"Male": 1, "Female": 0})

# One-hot encode race (keeping all categories for interpretability)
race_dummies = pd.get_dummies(df_clean["race"], prefix="race")
df_clean = pd.concat([df_clean, race_dummies], axis=1)

# Remove rows with any missing values
df_clean = df_clean.dropna()

print(f"Final dataset shape: {df_clean.shape}")
print(f"Readmission rate: {df_clean['readmission_flag'].mean():.3f}")

# =========================
# 4. Enhanced Exploratory Analysis
# =========================
print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Create subplots for better visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Overall readmission distribution
readmission_counts = df_clean["readmitted"].value_counts()
axes[0, 0].pie(readmission_counts.values, labels=readmission_counts.index, autopct='%1.1f%%')
axes[0, 0].set_title("Readmission Distribution")

# 2. Readmission by gender
gender_readmission = df_clean.groupby(["gender", "readmission_flag"]).size().unstack()
gender_readmission.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title("Readmission by Gender")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis='x', rotation=0)

# 3. Readmission by age group
age_readmission = df_clean.groupby(["age", "readmission_flag"]).size().unstack()
age_readmission.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title("Readmission by Age Group")
axes[1, 0].set_ylabel("Count")
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Correlation heatmap
numeric_cols = ["age_numeric", "gender_numeric", "time_in_hospital",
                "number_diagnoses", "readmission_flag"]
correlation_matrix = df_clean[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0, ax=axes[1, 1])
axes[1, 1].set_title("Correlation Matrix")

plt.tight_layout()
plt.show()

# Detailed statistics by demographics
print("\nReadmission rates by demographic groups:")
print("\nBy Gender:")
gender_stats = df_clean.groupby("gender")["readmission_flag"].agg(['count', 'mean', 'std'])
print(gender_stats)

print("\nBy Race:")
race_stats = df_clean.groupby("race")["readmission_flag"].agg(['count', 'mean', 'std'])
print(race_stats.sort_values('mean', ascending=False))

print("\nBy Age Group:")
age_stats = df_clean.groupby("age")["readmission_flag"].agg(['count', 'mean', 'std'])
print(age_stats.sort_values('mean', ascending=False))

# =========================
# 5. Enhanced Modeling
# =========================
print("\n" + "=" * 50)
print("PREDICTIVE MODELING")
print("=" * 50)

# Prepare features
feature_cols = ["age_numeric", "gender_numeric", "time_in_hospital", "number_diagnoses"] + \
               [col for col in df_clean.columns if col.startswith("race_")]

X = df_clean[feature_cols]
y = df_clean["readmission_flag"]

print(f"Features used: {feature_cols}")
print(f"Feature matrix shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# =========================
# 6. Enhanced Model Evaluation
# =========================
print("\nMODEL PERFORMANCE:")
print("=" * 30)

# Classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_score:.3f}")

# Confusion matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# =========================
# 7. Feature Importance Analysis
# =========================
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

# Create coefficient dataframe
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": log_reg.coef_[0],
    "Abs_Coefficient": np.abs(log_reg.coef_[0])
}).sort_values(by="Abs_Coefficient", ascending=False)

print("Feature Coefficients (Logistic Regression):")
print(coef_df)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df.head(10), x="Coefficient", y="Feature")
plt.title("Top 10 Feature Coefficients")
plt.xlabel("Logistic Regression Coefficient")
plt.tight_layout()
plt.show()

# =========================
# 8. Clinical Insights
# =========================
print("\n" + "=" * 50)
print("CLINICAL INSIGHTS")
print("=" * 50)

# Calculate odds ratios for interpretation
odds_ratios = np.exp(log_reg.coef_[0])
or_df = pd.DataFrame({
    "Feature": feature_cols,
    "Odds_Ratio": odds_ratios,
    "CI_Lower": np.exp(log_reg.coef_[0] - 1.96 * np.sqrt(np.diag(np.linalg.inv(X_train_scaled.T @ X_train_scaled)))),
    "CI_Upper": np.exp(log_reg.coef_[0] + 1.96 * np.sqrt(np.diag(np.linalg.inv(X_train_scaled.T @ X_train_scaled))))
}).sort_values(by="Odds_Ratio", ascending=False)

print("Odds Ratios (with 95% CI):")
for _, row in or_df.iterrows():
    feature = row["Feature"]
    or_val = row["Odds_Ratio"]
    ci_lower = row["CI_Lower"]
    ci_upper = row["CI_Upper"]

    if or_val > 1:
        direction = "increases"
        magnitude = f"{((or_val - 1) * 100):.1f}%"
    else:
        direction = "decreases"
        magnitude = f"{((1 - or_val) * 100):.1f}%"

    print(f"{feature}: OR = {or_val:.3f} [{ci_lower:.3f}-{ci_upper:.3f}]")
    print(f"  → {direction} readmission odds by {magnitude}")

# =========================
# 9. Save Enhanced Results
# =========================
print("\n" + "=" * 50)
print("SAVING RESULTS")
print("=" * 50)

# Comprehensive results dataframe
results_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": log_reg.coef_[0],
    "Odds_Ratio": odds_ratios,
    "Abs_Coefficient": np.abs(log_reg.coef_[0])
}).sort_values(by="Abs_Coefficient", ascending=False)

# Add model performance metrics
performance_metrics = {
    "AUC_ROC": auc_score,
    "Dataset_Size": len(df_clean),
    "Training_Size": len(X_train),
    "Test_Size": len(X_test),
    "Readmission_Rate": df_clean["readmission_flag"].mean()
}

# Save results
results_df.to_csv("enhanced_demographics_readmission_results.csv", index=False)
pd.DataFrame([performance_metrics]).to_csv("model_performance_metrics.csv", index=False)

print("Results saved to:")
print("- enhanced_demographics_readmission_results.csv")
print("- model_performance_metrics.csv")

print(f"\nModel Performance Summary:")
print(f"- AUC-ROC: {auc_score:.3f}")
print(f"- Dataset size: {len(df_clean):,} patients")
print(f"- Overall readmission rate: {df_clean['readmission_flag'].mean():.1%}")

# =========================
# 10. Clinical Recommendations
# =========================
print("\n" + "=" * 50)
print("CLINICAL RECOMMENDATIONS")
print("=" * 50)

print("Based on the demographic analysis:")

# Top risk factors
top_risks = results_df.head(3)
print("\nKey Risk Factors for Readmission:")
for _, row in top_risks.iterrows():
    feature = row["Feature"]
    or_val = row["Odds_Ratio"]
    if or_val > 1:
        print(f"• {feature}: {((or_val - 1) * 100):.1f}% increased odds")
    else:
        print(f"• {feature}: {((1 - or_val) * 100):.1f}% decreased odds")

print("\nClinical Action Items:")
print("• Consider enhanced discharge planning for high-risk demographic groups")
print("• Implement targeted follow-up protocols based on identified risk factors")
print("• Monitor length of stay and diagnosis complexity as key predictors")
print("• Develop culturally appropriate care pathways for different racial groups")

# Model limitations
print("\nModel Limitations:")
print("• Demographics alone may not capture full clinical complexity")
print("• Consider adding medication adherence, comorbidities, and social determinants")
print("• Temporal trends (1999-2008) may not reflect current healthcare landscape")
print(f"• Model explains limited variance (AUC = {auc_score:.3f})")

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
