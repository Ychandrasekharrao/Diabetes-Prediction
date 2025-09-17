# ==============================================================================
# SCRIPT: 02 - Model Training, Comparison, and Selection
# AUTHOR: [Your Name]
# DATE:   17-Sep-2025
# ==============================================================================

# ==================================================
# 1. SETUP AND IMPORTS
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib
import warnings
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Imblearn and SHAP
from imblearn.combine import SMOTETomek
import shap
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Add the 'src' directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.utils.data_loader import load_processed_data

# --- Config / Paths ---
sns.set(style="whitegrid")
REPORTS_DIR = PROJECT_ROOT / 'reports'
BEST_MODEL_FIG_DIR = REPORTS_DIR / 'figures' / 'best_model_analysis'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper function to save plots ---
def savefig(fig, name, tight=True):
    """Helper function to save figures to the reports directory."""
    p = BEST_MODEL_FIG_DIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved plot: {p.name}")

print("Model Training Script Started: Setup Complete.")

# ==================================================
# 2. LOAD PROCESSED DATA
# ==================================================
print("\n" + "="*50 + "\n2. Loading Processed Data\n" + "="*50)
df = load_processed_data()
if df.empty:
    sys.exit("Processed data not found. Please run the data processing script first.")

# ==================================================
# 3. PREPARE MODELING DATAFRAME
# ==================================================
print("\n" + "="*50 + "\n3. Preparing Data for Modeling\n" + "="*50)
X_model = df.drop(["target"], axis=1).copy()
if 'id' in X_model.columns: X_model = X_model.drop('id', axis=1)
y = df["target"].copy()

# --- Encoding ---
true_ordinal_cols = ["Cholesterol_Level", "Glucose_Level", "Smoking_Status", "Alcohol_Intake", "Physical_Activity", "BP_level"]
nominal_cols = ["Sex"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_model[true_ordinal_cols] = encoder.fit_transform(X_model[true_ordinal_cols])
X_model.rename(columns={col: f"ordinal_{col}" for col in true_ordinal_cols}, inplace=True)
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore")
encoded_features = ohe.fit_transform(X_model[nominal_cols])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(nominal_cols), index=X_model.index)
X_model = X_model.drop(nominal_cols, axis=1)
X_model = pd.concat([X_model, encoded_df], axis=1)
print("Feature encoding complete.")

# ==================================================
# 4. TRAIN-TEST SPLIT, SCALING, AND RESAMPLING
# ==================================================
print("\n" + "="*50 + "\n4. Splitting, Scaling, and Resampling Data\n" + "="*50)
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42, stratify=y)

# --- Feature Scaling ---
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
encoded_cols = [col for col in X_model.columns if 'ordinal_' in col or 'Sex_' in col]
cols_to_scale = [col for col in numeric_cols if col not in encoded_cols]
scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
print("Numerical features scaled successfully.")
print(f"Original training set distribution:\n{y_train.value_counts()}")

# --- Use the winning balancer from our experiment ---
balancer = SMOTETomek(random_state=42)
X_train_res, y_train_res = balancer.fit_resample(X_train, y_train)
print(f"\nResampled training set distribution (using SMOTE-Tomek):\n{y_train_res.value_counts()}")

# ==================================================
# 5. TRAIN AND EVALUATE MULTIPLE MODELS
# ==================================================
print("\n" + "="*50 + "\n5. Training and Evaluating Models\n" + "="*50)
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1),
    'XGBoost': XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
    'SVM': SVC(probability=True, random_state=42)
}
results = []
trained_models = {}
for name, model in models.items():
    print(f"--- Training {name} ---")
    model.fit(X_train_res, y_train_res)
    trained_models[name] = model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name, 'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred), 'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred), 'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    })

# ==================================================
# 6. COMPARE MODELS AND SELECT THE BEST
# ==================================================
print("\n" + "="*50 + "\n6. Model Comparison and Selection\n" + "="*50)
results_df = pd.DataFrame(results).sort_values(by=['ROC AUC', 'Recall'], ascending=False).reset_index(drop=True)
print("Model Performance Comparison Table:")
print(results_df)
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\n[WINNER] Best Model Selected (based on ROC AUC, then Recall): {best_model_name}")

# ==================================================
# 7. IN-DEPTH ANALYSIS OF THE BEST MODEL
# ==================================================
print("\n" + "="*50 + f"\n7. In-Depth Analysis of {best_model_name}\n" + "="*50)
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred_best)
print("Classification Report for the Best Model:")
print(classification_report(y_test, y_pred_best))

# --- Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
ax.set_title(f'Confusion Matrix for {best_model_name}')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
savefig(fig, '1_confusion_matrix.png')

# --- ROC AUC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
auc = roc_auc_score(y_test, y_pred_proba_best)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title(f'ROC Curve for {best_model_name}')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
savefig(fig, '2_roc_auc_curve.png')

# --- Predicted Probabilities Distribution ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(y_pred_proba_best[y_test==0], label='No Disease (Actual)', ax=ax, fill=True)
sns.kdeplot(y_pred_proba_best[y_test==1], label='Disease (Actual)', ax=ax, fill=True)
ax.set_title('Distribution of Predicted Probabilities by Actual Class')
ax.set_xlabel('Predicted Probability of Disease')
ax.set_ylabel('Density')
ax.legend()
savefig(fig, '3_predicted_probabilities.png')

# --- SHAP Analysis ---
print("Calculating SHAP values...")
if isinstance(best_model, (XGBClassifier, RandomForestClassifier, LGBMClassifier, AdaBoostClassifier)):
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
else:
    explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train_res, 100))
    shap_values = explainer.shap_values(X_test)[1]
    print("Using KernelExplainer (this may be slow)...")

# SHAP Summary Plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
plt.title(f"SHAP Summary for {best_model_name}")
# Use plt.gcf() to get the current figure that SHAP draws on
savefig(plt.gcf(), '4_shap_summary.png')

# SHAP Dependence Plot Grid
features_to_plot = ["ordinal_BP_level", "Pulse_Pressure", "Age_Years", "ordinal_Cholesterol_Level", "BMI", "ordinal_Glucose_Level"]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
shap_values_for_dep = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
for feature, ax in zip(features_to_plot, axes.flatten()):
    if feature in X_test.columns:
        shap.dependence_plot(feature, shap_values_for_dep, X_test, ax=ax, show=False)
fig.suptitle(f"SHAP Dependence Plots for {best_model_name}", fontsize=20, y=1.03)
savefig(fig, '5_shap_dependence_grid.png')

# ==================================================
# 8. SAVE THE BEST MODEL
# ==================================================
print("\n" + "="*50 + "\n8. Saving the Best Model\n" + "="*50)
model_path = REPORTS_DIR / f'best_model_{best_model_name.lower().replace(" ", "_")}.joblib'
joblib.dump(best_model, model_path)
print(f"✅ Best model saved to: {model_path}")

# ==================================================
# 9. GENERATE FINAL MARKDOWN REPORT
# ==================================================
print("\n" + "="*50 + "\n9. Generating Final Performance Report\n" + "="*50)
report_path = REPORTS_DIR / 'best_model_performance_report.md'

# --- Create relative paths for images for Markdown file ---
cm_path = Path.relative_to(BEST_MODEL_FIG_DIR / '1_confusion_matrix.png', REPORTS_DIR)
roc_path = Path.relative_to(BEST_MODEL_FIG_DIR / '2_roc_auc_curve.png', REPORTS_DIR)
prob_path = Path.relative_to(BEST_MODEL_FIG_DIR / '3_predicted_probabilities.png', REPORTS_DIR)
shap_summary_path = Path.relative_to(BEST_MODEL_FIG_DIR / '4_shap_summary.png', REPORTS_DIR)
shap_dep_path = Path.relative_to(BEST_MODEL_FIG_DIR / '5_shap_dependence_grid.png', REPORTS_DIR)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Model Performance Report\n\n")
    f.write(f"**Report generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**[WINNER] Best Performing Model (based on ROC AUC, then Recall):** `{best_model_name}`\n\n")
    f.write("## Model Comparison Table\n\n")
    f.write("```\n")
    f.write(results_df.to_string())
    f.write("\n```\n\n")
    f.write("---\n\n")
    f.write("## Detailed Analysis of Best Model\n\n")
    f.write("### Classification Report\n\n")
    f.write("```\n")
    f.write(classification_report(y_test, y_pred_best))
    f.write("\n```\n\n")
    f.write("### Confusion Matrix\n\n")
    f.write(f"![Confusion Matrix]({cm_path})\n\n")
    f.write("### ROC AUC Curve\n\n")
    f.write(f"![ROC AUC Curve]({roc_path})\n\n")
    f.write("### Predicted Probabilities Distribution\n\n")
    f.write(f"![Predicted Probabilities]({prob_path})\n\n")
    f.write("---\n\n")
    f.write("## Model Explainability (SHAP)\n\n")
    f.write("### SHAP Summary Plot\n\n")
    f.write(f"![SHAP Summary]({shap_summary_path})\n\n")
    f.write("### SHAP Dependence Plots\n\n")
    f.write(f"![SHAP Dependence Grid]({shap_dep_path})\n\n")

print(f"Markdown Performance report saved to: {report_path}")
print("\nModel Training Script Finished Successfully.")