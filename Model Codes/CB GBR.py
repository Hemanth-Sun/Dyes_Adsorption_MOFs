import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import zscore

from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

import shap
import warnings

warnings.filterwarnings("ignore")

# --- Define Output Directory ---
OUTPUT_DIR = r"/Users/hemanthsunkara/Downloads/outs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Load Dataset ---
df = pd.read_csv(os.path.join(OUTPUT_DIR, "/users/hemanthsunkara/downloads/ML_MO.csv"), encoding='latin1')

# --- Step 2: Remove duplicates by averaging uptake ---
df = df.groupby(df.columns.tolist()).mean().reset_index()

# --- Step 3: Handle sample names ---
sample_names = df.iloc[:, 0]  # Save for later
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

# --- Step 9: Hyperparameter Tuning ---

cat_param_grid = {
    'iterations':        [100, 200, 300, 500],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'depth':             [3, 5, 7, 10],
    'l2_leaf_reg':       [1, 3, 5, 7],
    'subsample':         [0.7, 0.8, 1.0],
    'colsample_bylevel': [0.7, 0.8, 1.0]
}

gbr_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 1.0]
}

cat_random = RandomizedSearchCV(
    CatBoostRegressor(random_state=42, verbose=0),
    cat_param_grid, n_iter=30, cv=5, scoring='r2',
    verbose=1, n_jobs=-1, random_state=42
)
cat_random.fit(X_train_scaled, y_train)
best_cat = cat_random.best_estimator_
print(f"Best CatBoost params: {cat_random.best_params_}")

gbr_random = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gbr_param_grid, n_iter=30, cv=5, scoring='r2',
    verbose=1, n_jobs=-1, random_state=42
)
gbr_random.fit(X_train_scaled, y_train)
best_gbr = gbr_random.best_estimator_
print(f"Best GradientBoosting params: {gbr_random.best_params_}")

# --- Step 10: Cross-Validation Scores for Base Models ---
for model_name, model, X_data in [('CatBoost', best_cat, X_train_scaled),
                                   ('GradientBoosting', best_gbr, X_train_scaled)]:
    cv_rmse = cross_val_score(model, X_data, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_r2 = cross_val_score(model, X_data, y_train, cv=5, scoring='r2')
    print(f"\n{model_name} CV R²: {cv_r2}")
    print(f"{model_name} CV RMSE: {np.abs(cv_rmse)}")
    print(f"{model_name} Mean CV RMSE: {np.mean(np.abs(cv_rmse)):.3f}")

# --- Step 11: Define Stacking Regressor ---
estimators = [('cat', best_cat), ('gbr', best_gbr)]
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),  # meta model
    cv=5, n_jobs=-1, passthrough=False
)
stacking_regressor.fit(X_train_scaled, y_train)

# --- Step 12: Predictions ---
y_pred_stack = stacking_regressor.predict(X_test_scaled)
y_pred_train_stack = stacking_regressor.predict(X_train_scaled)


# --- Step 13: Metrics ---
def display_metrics(y_true, y_pred, dataset="Test"):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

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
    print(f"  MAE    : {mae:.4f}")
    print(f"  MAPE   : {mape:.4f} %")
    print(f"  sMAPE  : {smape:.4f} %")
    print(f"  MSLE   : {msle:.6f}")
    print(f"  RMSLE  : {rmsle:.6f}")
    print(f"  NRMSE  : {nrmse:.6f}")

    return dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape, sMAPE=smape, MSLE=msle, RMSLE=rmsle, NRMSE=nrmse)


test_metrics = display_metrics(y_test, y_pred_stack, "Test")
train_metrics = display_metrics(y_train, y_pred_train_stack, "Training")

# --- Step 14: Plot Actual vs Predicted ---
plt.figure(figsize=(9, 7))
plt.scatter(y_test, y_pred_stack, c='#1E90FF', edgecolors='black', linewidths=1, s=80, label='Test Data', alpha=0.85)
plt.scatter(y_train, y_pred_train_stack, c='#FF3030', edgecolors='black', linewidths=1, s=80, label='Train Data',
            alpha=0.85)

min_val = min(min(y_test), min(y_pred_stack), min(y_train), min(y_pred_train_stack))
max_val = max(max(y_test), max(y_pred_stack), max(y_train), max(y_pred_train_stack))
plt.plot([min_val, max_val], [min_val, max_val], color='#FFD700', linestyle='--', linewidth=2.5, label='Ideal (y = x)')

plt.xlabel("Measured Adsorption Capacity (mg/g)", fontsize=14, weight='bold')
plt.ylabel("Predicted Adsorption Capacity (mg/g)", fontsize=14, weight='bold')
plt.title("Stacking Ensemble (CB + GBR): Adsorption Capacity Prediction", fontsize=15, weight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black')
plt.tight_layout()
plt.show()

# --- Step 15: SHAP Analysis for CatBoost and GradientBoosting ---

# CatBoost Explainer
explainer_cat = shap.TreeExplainer(best_cat)
shap_values_cat = explainer_cat.shap_values(X_test_scaled)

# GradientBoosting Explainer
explainer_gbr = shap.TreeExplainer(best_gbr)
shap_values_gbr = explainer_gbr.shap_values(X_test_scaled)

# Average SHAP values across both base learners
shap_values_avg = (shap_values_cat + shap_values_gbr) / 2

# SHAP summary plot (averaged)
shap.summary_plot(shap_values_avg, X_test_scaled, feature_names=X.columns, show=False)
plt.title("SHAP Summary Plot - Stacking (CatBoost + GBR)", fontsize=13, weight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "Stacking_CB_GBR_SHAP_summary.png"), dpi=300, bbox_inches="tight")
plt.close()

# SHAP bar plot (averaged)
shap.summary_plot(shap_values_avg, X_test_scaled, feature_names=X.columns, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Stacking (CatBoost + GBR)", fontsize=13, weight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "Stacking_CB_GBR_SHAP_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# --- Step 16: Build DataFrames for Excel export ---
actuals_test_df = pd.DataFrame({
    "Actual_Adsorption_Capacity_mg_g": y_test.values,
    "Predicted_Adsorption_Capacity_mg_g": y_pred_stack
})

actuals_train_df = pd.DataFrame({
    "Actual_Adsorption_Capacity_mg_g": y_train.values,
    "Predicted_Adsorption_Capacity_mg_g": y_pred_train_stack
})

shap_cat_df = pd.DataFrame(shap_values_cat, columns=X.columns)
shap_gbr_df = pd.DataFrame(shap_values_gbr, columns=X.columns)
shap_avg_df = pd.DataFrame(shap_values_avg, columns=X.columns)

metrics_df = pd.DataFrame({
    "Metric": ["R²", "RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "MSLE", "RMSLE", "NRMSE"],
    "Test Set": [
        test_metrics["R2"], test_metrics["RMSE"], test_metrics["MAE"], test_metrics["MAPE"],
        test_metrics["sMAPE"], test_metrics["MSLE"], test_metrics["RMSLE"],
        test_metrics["NRMSE"]
    ],
    "Training Set": [
        train_metrics["R2"], train_metrics["RMSE"], train_metrics["MAE"], train_metrics["MAPE"],
        train_metrics["sMAPE"], train_metrics["MSLE"], train_metrics["RMSLE"],
        train_metrics["NRMSE"]
    ]
})

# --- Step 17: Save to single Excel file (multiple sheets) ---
output_path = os.path.join(OUTPUT_DIR, "Stacking_CB_GBR_Results.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    actuals_test_df.to_excel(writer, sheet_name="Actual_vs_Predicted", index=False)
    actuals_train_df.to_excel(writer, sheet_name="Actual_vs_Predicted_Train", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    shap_avg_df.to_excel(writer, sheet_name="SHAP_Averaged", index=False)
    shap_cat_df.to_excel(writer, sheet_name="SHAP_CatBoost", index=False)
    shap_gbr_df.to_excel(writer, sheet_name="SHAP_GradientBoosting", index=False)

print(f"\nAll results saved to '{output_path}'")
print(
    "  Sheets: Actual_vs_Predicted | Actual_vs_Predicted_Train | Metrics | SHAP_Averaged | SHAP_CatBoost | SHAP_GradientBoosting")


# --- Step 18: Regression Error Characteristic (REC) Curve ---
def plot_rec_curve(y_true, y_pred, model_name="Model"):
    errors = np.abs(y_true - y_pred)
    max_error = np.max(errors)
    eps = np.linspace(0, max_error, 1000)
    accuracy = [np.mean(errors <= e) for e in eps]

    plt.figure(figsize=(8, 6))
    plt.plot(eps, accuracy, label=f"{model_name} REC", color="green", linewidth=2)
    plt.xlabel("Absolute Error Tolerance", fontsize=13, weight="bold")
    plt.ylabel("Fraction of Samples within Tolerance", fontsize=13, weight="bold")
    plt.title(f"Regression Error Characteristic (REC) Curve - {model_name}", fontsize=14, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    rec_path = os.path.join(OUTPUT_DIR, f"{model_name}_REC_curve.png")
    plt.savefig(rec_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"REC curve saved as {rec_path}")


plot_rec_curve(y_test, y_pred_stack, model_name="Stacking_Ensemble_CB_GBR")
