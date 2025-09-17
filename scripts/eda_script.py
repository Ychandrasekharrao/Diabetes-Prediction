# ==================================================
# 1. SETUP AND IMPORTS
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Add the 'src' directory to the Python path to allow for utility imports
# This assumes the script is run from the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.utils.data_loader import save_processed_data

# --- Plotting Style and Directories ---
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
FIG_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'eda'
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(fig, name, tight=True):
    """Helper function to save figures to the reports directory."""
    p = FIG_DIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig) # Close figure to free up memory

print("EDA Script Started: Setup Complete.")
print(f"Plots will be saved to: {FIG_DIR}")

# ==================================================
# 2. DATA LOADING AND INITIAL INSPECTION
# ==================================================
print("\n" + "="*50)
print("2. Data Loading and Initial Inspection")
print("="*50)

raw_path = PROJECT_ROOT / "data" / "raw" / "heart disease.csv"
df = pd.read_csv(raw_path)
df = df.rename(columns={
    'age': 'Age', 'gender': 'Sex', 'height': 'Height', 'weight': 'Weight',
    'ap_hi': 'Systolic_BP', 'ap_lo': 'Diastolic_BP',
    'cholesterol': 'Cholesterol_Level', 'gluc': 'Glucose_Level',
    'smoke': 'Smoking_Status', 'alco': 'Alcohol_Intake',
    'active': 'Physical_Activity', 'cardio': 'target'
})
print(f"Initial dataset shape: {df.shape}")

# ==================================================
# 3. DATA CLEANING AND PREPROCESSING
# ==================================================
print("\n" + "="*50)
print("3. Data Cleaning and Preprocessing")
print("="*50)

df['Age_Years'] = round(df['Age'] / 365, 0)
df.drop('Age', axis=1, inplace=True)
df['Height_mt'] = df['Height'] / 100
df.drop('Height', axis=1, inplace=True)

rows_before = len(df)
df = df[
    (df['Systolic_BP'] >= 90) & (df['Systolic_BP'] <= 250) &
    (df['Diastolic_BP'] >= 60) & (df['Diastolic_BP'] <= 150) &
    (df['Diastolic_BP'] <= df['Systolic_BP'])
].copy()
print(f"Removed {rows_before - len(df)} rows due to invalid blood pressure readings.")

rows_before = len(df)
df['BMI'] = df['Weight'] / (df['Height_mt'] ** 2)
df = df[
    (df['Height_mt'].between(1.3, 2.1)) &
    (df['BMI'].between(15, 60))
].copy()
print(f"Removed {rows_before - len(df)} rows due to unrealistic Height or BMI.")
print(f"Final cleaned dataset shape: {df.shape}")

# ==================================================
# 4. FEATURE ENGINEERING
# ==================================================
print("\n" + "="*50)
print("4. Feature Engineering")
print("="*50)

df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']
def classify_bp(row):
    systolic, diastolic = row['Systolic_BP'], row['Diastolic_BP']
    if systolic < 120 and diastolic < 80: return 'Normal'
    elif 120 <= systolic <= 129 and diastolic < 80: return 'Elevated'
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89: return 'Hypertension Stage 1'
    elif 140 <= systolic <= 180 or 90 <= diastolic <= 120: return 'Hypertension Stage 2'
    else: return 'Hypertensive Crisis'
df['BP_level'] = df.apply(classify_bp, axis=1)

age_bins = [29, 40, 50, 60, 70]
age_labels = ['30-40', '41-50', '51-60', '61-70']
df['Age_Group'] = pd.cut(df['Age_Years'], bins=age_bins, labels=age_labels, right=True)

bmi_bins = [0, 18.5, 25, 30, 35, np.inf]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II+']
df['BMI_Category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)
print("Created 'Pulse_Pressure', 'BP_level', 'Age_Group', and 'BMI_Category' features.")

# ==================================================
# 5. COMPREHENSIVE EXPLORATORY DATA ANALYSIS
# ==================================================
print("\n" + "="*50)
print("5. Generating and Saving Comprehensive EDA Plots")
print("="*50)

df['target_name'] = df['target'].map({0: 'No Disease', 1: 'Disease'})
palette = {'No Disease':'#56B4E9', 'Disease':'#E69F00'}

# --- 5.A. Target Distribution ---
fig, ax = plt.subplots()
sns.countplot(x='target_name', data=df, ax=ax, palette=palette, hue='target_name', legend=False)
ax.set_title('Target Variable Distribution')
savefig(fig, '01_target_distribution.png')

# --- 5.B. Univariate Numerical Analysis ---
numeric_cols = ['Age_Years', 'Weight', 'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure', 'BMI']
fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 4 * len(numeric_cols)))
fig.suptitle('Univariate Analysis of Numerical Features', fontsize=16, y=1.02)
for i, col in enumerate(numeric_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i, 0], hue='target_name', palette=palette)
    axes[i, 0].set_title(f'{col} Distribution')
    sns.boxplot(data=df, x='target_name', y=col, ax=axes[i, 1], palette=palette)
    axes[i, 1].set_title(f'{col} by Target')
savefig(fig, '02_univariate_numerical_analysis.png')

# --- 5.C. Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(12, 8))
corr = df[numeric_cols + ['target']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features')
savefig(fig, '03_correlation_heatmap.png')

# --- 5.D. Pair Plot on a Stratified Sample ---
pairgrid_cols = ['BMI','Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure', 'Age_Years']
df_sample, _ = train_test_split(df, train_size=min(5000, len(df)), stratify=df['target'], random_state=42)
g = sns.pairplot(df_sample, vars=pairgrid_cols, hue='target_name', palette=palette, corner=True)
g.fig.suptitle('Pairwise Relationships (Stratified Sample)', fontsize=16, y=1.03)
savefig(g.fig, '04_pairwise_numerical_analysis.png')

# --- 5.E. Categorical Analysis Grids ---
categorical_groups = {
    'Demographics': ['Sex', 'Age_Group', 'BMI_Category'],
    'Vitals_Biochemical': ['BP_level', 'Cholesterol_Level', 'Glucose_Level'],
    'Lifestyle': ['Smoking_Status', 'Alcohol_Intake', 'Physical_Activity']
}
for group_name, cols in categorical_groups.items():
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(5*n, 5*n), squeeze=False)
    fig.suptitle(f'Categorical Analysis Grid: {group_name}', fontsize=16, y=1.02)
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ct = pd.crosstab(df[col1], df['target_name'], normalize='index') * 100
                ct.plot(kind='bar', stacked=True, color=[palette['No Disease'], palette['Disease']], ax=ax, rot=0)
                ax.set_title(f'Target Distribution by {col1}')
                ax.set_ylabel('Percentage')
                ax.legend(title='Target')
            else:
                ct = pd.crosstab(df[col1], df[col2])
                sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                ax.set_title(f'{col1} vs {col2}')
    savefig(fig, f'05_categorical_grid_{group_name.lower()}.png')

df = df.drop('target_name', axis=1)
print("Generated all EDA plots.")

# ==================================================
# 6. SAVE PROCESSED DATA FOR MODELING
# ==================================================
print("\n" + "="*50)
print("6. Saving Processed Data")
print("="*50)

# Drop columns not needed for the final model (e.g., intermediate or high-level categorical groups)
final_df = df.drop(['Height_mt', 'Weight', 'Age_Group', 'BMI_Category'], axis=1)
save_processed_data(final_df, filename="processed_heart_disease.csv")

print("\nEDA and Preprocessing Script Finished Successfully.")