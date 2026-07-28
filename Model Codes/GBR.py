import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler

import shap
import warnings

warnings.filterwarnings("ignore")

## --- Define Output Directory ---
OUTPUT_DIR = r"/users/hemanthsunkara/downloads/outs/GBR/MO"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Load Dataset ---
df = pd.read_csv("/users/hemanthsunkara/downloads/ML_MO.csv", encoding='latin1')

# --- Step 2: Remove duplicates by averaging uptake ---
df = df.groupby(df.columns.tolist()).mean().reset_index()

# --- Step 3: Handle sample names ---
sample_names = df.iloc[:, 0]
df = df.iloc[:, 1:]

# --- Step 4: Check for missing values ---
assert not df.isnull().values.any(), "Missing values detected!"

# --- Step 5: Outlier detection ---
z_scores = df.select_dtypes(include=np.number).apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
print(f"Outliers found: {outliers.sum()}")

# --- Step 6: Correlation heatmap ---
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), cmap='seismic', annot=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- Step 7: Train/Test Split ---
X = df.drop(columns=["Adsorption Capacity (mg/g)"])
y = df["Adsorption Capacity (mg/g)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 8: Scaling ---
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Step 9: Metrics Suite ---
def compute_metrics(y_true, y_pred, dataset="Test"):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE — guard against zero actuals
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # sMAPE
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100

    # MSLE / RMSLE — clip negatives to avoid log(negative)
    y_true_c = np.clip(y_true, 0, None)
    y_pred_c = np.clip(y_pred, 0, None)
    msle = np.mean((np.log1p(y_true_c) - np.log1p(y_pred_c)) ** 2)
    rmsle = np.sqrt(msle)

    # NRMSE (normalised by range)
    nrmse = rmse / (y_true.max() - y_true.min()) if y_true.max() != y_true.min() else np.nan

    print(f"\n{dataset} Set Metrics:")
    print(f"  R²     : {r2:.4f}")
    print(f"  RMSE   : {rmse:.4f}")
    print(f"  MAPE   : {mape:.4f} %")
    print(f"  sMAPE  : {smape:.4f} %")
    print(f"  MSLE   : {msle:.6f}")
    print(f"  RMSLE  : {rmsle:.6f}")
    print(f"  NRMSE  : {nrmse:.6f}")

    return dict(R2=r2, RMSE=rmse, MAPE=mape, sMAPE=smape, MSLE=msle, RMSLE=rmsle, NRMSE=nrmse)


# --- Step 10: Hyperparameter Tuning ---
gbr_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 1.0]
}

gbr_random = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gbr_param_grid, n_iter=30, cv=5, scoring='r2',
    verbose=1, n_jobs=-1, random_state=42
)
gbr_random.fit(X_train_scaled, y_train)
best_gbr = gbr_random.best_estimator_
print(f"Best GradientBoosting params: {gbr_random.best_params_}")

# --- Step 11: Cross-Validation Scores ---
cv_rmse = cross_val_score(best_gbr, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_r2 = cross_val_score(best_gbr, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nGradientBoosting CV R²  : {cv_r2}")
print(f"GradientBoosting CV RMSE: {np.abs(cv_rmse)}")
print(f"GradientBoosting Mean CV RMSE: {np.mean(np.abs(cv_rmse)):.3f}")

# --- Step 12: Fit GradientBoosting Model ---
best_gbr.fit(X_train_scaled, y_train)

# --- Step 13: Predictions ---
y_pred_test = best_gbr.predict(X_test_scaled)
y_pred_train = best_gbr.predict(X_train_scaled)

# --- Step 14: Metrics ---
test_metrics = compute_metrics(y_test, y_pred_test, dataset="Test")
train_metrics = compute_metrics(y_train, y_pred_train, dataset="Training")

# --- Step 15: Plot Actual vs Predicted ---
plt.figure(figsize=(9, 7))
plt.scatter(y_test, y_pred_test, c='#1E90FF', edgecolors='black', linewidths=1,
            s=80, label='Test Data', alpha=0.85)
plt.scatter(y_train, y_pred_train, c='#FF3030', edgecolors='black', linewidths=1,
            s=80, label='Train Data', alpha=0.85)

all_vals = np.concatenate([y_test, y_pred_test, y_train, y_pred_train])
min_val, max_val = all_vals.min(), all_vals.max()
plt.plot([min_val, max_val], [min_val, max_val],
         color='#FFD700', linestyle='--', linewidth=2.5, label='Ideal (y = x)')

plt.xlabel("Measured Adsorption Capacity (mg/g)", fontsize=14, weight='bold')
plt.ylabel("Predicted Adsorption Capacity (mg/g)", fontsize=14, weight='bold')
plt.title("GradientBoosting: Adsorption Capacity Prediction", fontsize=14, weight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black')
plt.tight_layout()
plt.show()

# --- Step 16: SHAP Analysis for GradientBoosting ---
explainer_gbr = shap.TreeExplainer(best_gbr)
shap_values_gbr = explainer_gbr.shap_values(X_test_scaled)

# SHAP summary plot
shap.summary_plot(shap_values_gbr, X_test_scaled, feature_names=X.columns, show=False)
plt.title("SHAP Summary Plot - GradientBoosting", fontsize=13, weight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "GradientBoosting_SHAP_summary.png"), dpi=300, bbox_inches="tight")
plt.close()

# SHAP bar plot
shap.summary_plot(shap_values_gbr, X_test_scaled, feature_names=X.columns,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance - GradientBoosting", fontsize=13, weight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "GradientBoosting_SHAP_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# --- Step 17: Build DataFrames for Excel export ---
actuals_test_df = pd.DataFrame({
    "Actual_Adsorption_Capacity_mg_g": y_test.values,
    "Predicted_Adsorption_Capacity_mg_g": y_pred_test
})

actuals_train_df = pd.DataFrame({
    "Actual_Adsorption_Capacity_mg_g": y_train.values,
    "Predicted_Adsorption_Capacity_mg_g": y_pred_train
})

shap_gbr_df = pd.DataFrame(shap_values_gbr, columns=X.columns)

metrics_df = pd.DataFrame({
    "Metric": ["R²", "RMSE", "MAPE (%)", "sMAPE (%)", "MSLE", "RMSLE", "NRMSE"],
    "Test Set": [
        test_metrics["R2"], test_metrics["RMSE"], test_metrics["MAPE"],
        test_metrics["sMAPE"], test_metrics["MSLE"], test_metrics["RMSLE"],
        test_metrics["NRMSE"]
    ],
    "Training Set": [
        train_metrics["R2"], train_metrics["RMSE"], train_metrics["MAPE"],
        train_metrics["sMAPE"], train_metrics["MSLE"], train_metrics["RMSLE"],
        train_metrics["NRMSE"]
    ]
})

# --- Step 18: Save to single Excel file (multiple sheets) ---
output_path = os.path.join(OUTPUT_DIR, "GradientBoosting_Results.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    actuals_test_df.to_excel(writer, sheet_name="Actual_vs_Predicted", index=False)
    actuals_train_df.to_excel(writer, sheet_name="Actual_vs_Predicted_Train", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    shap_gbr_df.to_excel(writer, sheet_name="SHAP_GradientBoosting", index=False)

print(f"\nAll results saved to '{output_path}'")
print("  Sheets: Actual_vs_Predicted | Actual_vs_Predicted_Train | Metrics | SHAP_GradientBoosting")


# --- Step 19: REC Curve ---
def plot_rec_curve(y_true, y_pred, model_name="Model", color="green"):
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    eps = np.linspace(0, errors.max(), 1000)
    accuracy = [np.mean(errors <= e) for e in eps]

    plt.figure(figsize=(8, 6))
    plt.plot(eps, accuracy, label=f"{model_name} REC", color=color, linewidth=2)
    plt.xlabel("Absolute Error Tolerance", fontsize=13, weight="bold")
    plt.ylabel("Fraction of Samples within Tolerance", fontsize=13, weight="bold")
    plt.title(f"Regression Error Characteristic (REC) Curve - {model_name}",
              fontsize=14, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    rec_path = os.path.join(OUTPUT_DIR, f"{model_name}_REC_curve.png")
    plt.savefig(rec_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"REC curve saved as {rec_path}")


plot_rec_curve(y_test, y_pred_test, model_name="GradientBoosting", color="green")