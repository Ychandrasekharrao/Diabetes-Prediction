# ==============================================================================
# SCRIPT: 03 - Resampling Technique Comparison
# AUTHOR: [Your Name]
# DATE:   17-Sep-2025
# ==============================================================================

# ==================================================
# 1. SETUP AND IMPORTS
# ==================================================
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, recall_score, confusion_matrix

# Imblearn for resampling techniques
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

# A fast model for quick comparison
from lightgbm import LGBMClassifier

# Add the 'src' directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.utils.data_loader import load_processed_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
print("Balancer Comparison Script Started: Setup Complete.")

# ==================================================
# 2. LOAD AND PREPARE DATA
# ==================================================
print("\n" + "="*50 + "\n2. Loading and Preparing Data\n" + "="*50)
df = load_processed_data()
if df.empty:
    sys.exit("Processed data not found. Please run the data processing script first.")

X_model = df.drop(["target"], axis=1).copy()
if 'id' in X_model.columns: X_model = X_model.drop('id', axis=1)
y = df["target"].copy()

# --- Encoding ---
true_ordinal_cols = ["Cholesterol_Level", "Glucose_Level", "Smoking_Status", "Alcohol_Intake", "Physical_Activity", "BP_level"]
nominal_cols = ["Sex"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_model[true_ordinal_cols] = encoder.fit_transform(X_model[true_ordinal_cols])
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore")
encoded_features = ohe.fit_transform(X_model[nominal_cols])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(nominal_cols), index=X_model.index)
X_model = X_model.drop(nominal_cols, axis=1)
X_model = pd.concat([X_model, encoded_df], axis=1)
print("Feature encoding complete.")

# --- Train-Test Split & Scaling ---
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42, stratify=y)
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
encoded_cols = [col for col in X_model.columns if 'ordinal_' in col or 'Sex_' in col]
cols_to_scale = [col for col in numeric_cols if col not in encoded_cols]
scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
print("Data splitting and scaling complete.")

# ==================================================
# 3. COMPARE BALANCING TECHNIQUES
# ==================================================
print("\n" + "="*50 + "\n3. Comparing Balancing Techniques\n" + "="*50)

balancers = {
    'Original (Imbalanced)': None,
    'SMOTE (Oversampling)': SMOTE(random_state=42),
    'ADASYN (Oversampling)': ADASYN(random_state=42),
    'SMOTE-ENN (Combined)': SMOTEENN(random_state=42),
    'SMOTE-Tomek (Combined)': SMOTETomek(random_state=42)
}

results = []
model_for_testing = LGBMClassifier(random_state=42, verbosity=-1)

for name, balancer in balancers.items():
    print(f"--- Testing Balancer: {name} ---")
    
    try:
        # Apply the balancer
        if balancer is None:
            X_res, y_res = X_train.copy(), y_train.copy()
        else:
            X_res, y_res = balancer.fit_resample(X_train, y_train)
            
        # Train the model
        model_for_testing.fit(X_res, y_res)
        
        # Make predictions on the test set
        y_pred = model_for_testing.predict(X_test)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) # False Positive Rate
        fnr = fn / (fn + tp) # False Negative Rate (1 - Recall)
        
        results.append({
            'Balancer': name,
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'FPR': fpr,
            'FNR': fnr
        })
    except Exception as e:
        print(f"    -> ⚠️ Could not apply {name}. Reason: {e}")
        print(f"    -> Skipping to next balancer.")
        continue # Skip to the next iteration of the loop

# ==================================================
# 4. SHOW RESULTS
# ==================================================
print("\n" + "="*50 + "\n4. Balancer Performance Comparison\n" + "="*50)

results_df = pd.DataFrame(results).sort_values(by='Recall', ascending=False).reset_index(drop=True)
print(results_df)

print("\nBalancer Comparison Script Finished.")