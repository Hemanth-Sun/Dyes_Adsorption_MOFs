import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import zscore

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler

import shap
import warnings

warnings.filterwarnings("ignore")

# --- Step 1: Load Dataset ---
df = pd.read_csv(r"/users/hemanthsunkara/downloads/ML_MO.csv", encoding='latin1')
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

# --- Step 2: Remove duplicates by averaging uptake ---
df = df.groupby(df.columns.tolist()).mean().reset_index()

# --- Step 3: Handle sample names (Group Identifier) ---
sample_names = df.iloc[:, 0].reset_index(drop=True)
df = df.iloc[:, 1:].reset_index(drop=True)

n_unique_mofs = sample_names.nunique()
print(f"Total rows: {len(df)} | Unique MOFs (groups): {n_unique_mofs}")

# --- Step 4: Check for missing values ---
assert not df.isnull().values.any(), "Missing values detected!"

target_col = "Adsorption Capacity (mg/g)"
X = df.drop(columns=[target_col])
y = df[target_col]
groups = sample_names


# --- Step 5: Corrected Kennard-Stone (Pre-Scaled Centroid Space) ---
def kennard_stone_group_split(X_df, group_series, train_ratio=0.80):
    unique_groups = group_series.unique()
    n_groups = len(unique_groups)
    n_train_groups = int(np.round(n_groups * train_ratio))

    scaler_ks = StandardScaler()
    X_scaled = scaler_ks.fit_transform(X_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns, index=X_df.index)

    group_centroids = []
    for g in unique_groups:
        centroid = X_scaled_df[group_series == g].mean(axis=0).values
        group_centroids.append(centroid)

    group_centroids = np.array(group_centroids)
    dist_matrix = cdist(group_centroids, group_centroids, metric='euclidean')

    i1, i2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected_indices = [i1, i2]
    remaining_indices = list(set(range(n_groups)) - set(selected_indices))

    while len(selected_indices) < n_train_groups:
        sub_dist = dist_matrix[remaining_indices][:, selected_indices]
        min_dists = np.min(sub_dist, axis=1)
        next_choice = remaining_indices[np.argmax(min_dists)]
        selected_indices.append(next_choice)
        remaining_indices.remove(next_choice)

    train_groups = set(unique_groups[selected_indices])
    test_groups = set(unique_groups[remaining_indices])

    train_mask = group_series.isin(train_groups)
    test_mask = group_series.isin(test_groups)

    return train_mask, test_mask, train_groups, test_groups


train_mask, test_mask, train_mofs, test_mofs = kennard_stone_group_split(X, groups, train_ratio=0.80)

X_train, X_test = X[train_mask].reset_index(drop=True), X[test_mask].reset_index(drop=True)
y_train, y_test = y[train_mask].reset_index(drop=True), y[test_mask].reset_index(drop=True)
groups_train = groups[train_mask].reset_index(drop=True)
groups_test = groups[test_mask].reset_index(drop=True)

# Verification
leak_check = set(groups_train) & set(groups_test)
assert len(leak_check) == 0, f"Data leakage detected! Overlapping MOFs: {leak_check}"

train_pct = (len(X_train) / len(df)) * 100
test_pct = (len(X_test) / len(df)) * 100
print("\n" + "=" * 60)
print("SPACE-FILLING (KENNARD-STONE) TRUE BLIND SPLIT:")
print(f"  Train/Val : {len(X_train)} rows ({train_pct:.1f}%) across {groups_train.nunique()} MOFs")
print(f"  Blind Test: {len(X_test)} rows ({test_pct:.1f}%) across {groups_test.nunique()} MOFs")
print("  Test MOFs held out:", sorted(list(test_mofs)))
print("=" * 60 + "\n")

# --- Step 6: Post-Split Diagnostics (Zero Leakage) ---
z_scores = X_train.select_dtypes(include=np.number).apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
print(f"Training Outliers found: {outliers.sum()} (Diagnostic only)")


# --- Step 7: Feature Scaling & Structure Locking ---
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)


# --- Step 8: Anchor-Preserved Group CV Generator ---
def build_anchor_preserved_cv(y_tr, g_tr, n_splits=5):
    mof_max_caps = y_tr.groupby(g_tr).max()
    anchor_top = mof_max_caps.idxmax()
    anchor_bottom = mof_max_caps.idxmin()
    anchor_mofs = [anchor_top, anchor_bottom]

    unique_mofs = g_tr.unique()
    remaining_mofs = [m for m in unique_mofs if m not in anchor_mofs]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = []
    g_tr_arr = g_tr.values

    for tr_m_idx, val_m_idx in kf.split(remaining_mofs):
        val_mof_names = [remaining_mofs[i] for i in val_m_idx]
        val_idx = np.where(np.isin(g_tr_arr, val_mof_names))[0]
        tr_idx = np.where(~np.isin(g_tr_arr, val_mof_names))[0]
        cv_splits.append((tr_idx, val_idx))

    return cv_splits, anchor_mofs


anchor_cv_splits, anchor_mofs = build_anchor_preserved_cv(y_train, groups_train, n_splits=5)
print(f"Locked Anchor MOFs (Internal CV): {anchor_mofs}")


# --- Step 9: Metrics Suite ---
def compute_metrics(y_true, y_pred, dataset="Test"):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100

    y_true_c = np.clip(y_true, 0, None)
    y_pred_c = np.clip(y_pred, 0, None)
    msle = np.mean((np.log1p(y_true_c) - np.log1p(y_pred_c)) ** 2)
    rmsle = np.sqrt(msle)

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


# --- Step 10: Group-Aware Hyperparameter Tuning (XGBoost) ---
xgb_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'reg_lambda': [1, 5, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    xgb_param_grid, n_iter=30,
    cv=anchor_cv_splits,
    scoring='r2', verbose=1, n_jobs=-1, random_state=42
)
xgb_random.fit(X_train_scaled, y_train)
best_xgb = xgb_random.best_estimator_
print(f"\nBest Tuned XGBoost params: {xgb_random.best_params_}")

# --- Step 11: Final Training Fit & Predictions ---
best_xgb.fit(X_train_scaled, y_train)

y_pred_test = best_xgb.predict(X_test_scaled)
y_pred_train = best_xgb.predict(X_train_scaled)

print("\n" + "=" * 60)
test_metrics = compute_metrics(y_test, y_pred_test, dataset="Blind Test (Unseen MOFs)")
train_metrics = compute_metrics(y_train, y_pred_train, dataset="Training")
print("=" * 60)

# --- Step 12: Actual vs Predicted Scatter Plot ---
plt.figure(figsize=(9, 7))
plt.scatter(y_test, y_pred_test, c='#1E90FF', edgecolors='black', linewidths=1,
            s=80, label='Blind Test Data (unseen MOFs)', alpha=0.85)
plt.scatter(y_train, y_pred_train, c='#FF3030', edgecolors='black', linewidths=1,
            s=80, label='Train Data', alpha=0.85)

all_vals = np.concatenate([y_test, y_pred_test, y_train, y_pred_train])
min_val, max_val = all_vals.min(), all_vals.max()
plt.plot([min_val, max_val], [min_val, max_val],
         color='#FFD700', linestyle='--', linewidth=2.5, label='Ideal (y = x)')

plt.xlabel("Measured Adsorption Capacity (mg/g)", fontsize=14, weight='bold')
plt.ylabel("Predicted Adsorption Capacity (mg/g)", fontsize=14, weight='bold')
plt.title("XGBoost: Adsorption Capacity Prediction\n(Rigorous 3-Way Split Protocol)",
          fontsize=13, weight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig("XGBoost_Actual_vs_Predicted.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Step 13: Physically Grounded SHAP Plots (Numpy Arrays Passed) ---
explainer_xgb = shap.TreeExplainer(best_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test_scaled)

# Summary plot
shap.summary_plot(
    shap_values_xgb,
    X_test.values,
    feature_names=X_test.columns,
    show=False
)
plt.title("SHAP Summary Plot - Physically Grounded Units", fontsize=13, weight='bold')
plt.savefig("XGBoost_SHAP_summary.png", dpi=300, bbox_inches="tight")
plt.show()

# Bar plot
shap.summary_plot(
    shap_values_xgb,
    X_test.values,
    feature_names=X_test.columns,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance", fontsize=13, weight='bold')
plt.savefig("XGBoost_SHAP_bar.png", dpi=300, bbox_inches="tight")
plt.show()


# --- Step 14: REC Curve ---
def plot_rec_curve(y_true, y_pred, model_name="XGBoost", color="red"):
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    eps = np.linspace(0, errors.max(), 1000)
    accuracy = [np.mean(errors <= e) for e in eps]

    plt.figure(figsize=(8, 6))
    plt.plot(eps, accuracy, label=f"{model_name} REC", color=color, linewidth=2)
    plt.xlabel("Absolute Error Tolerance (mg/g)", fontsize=13, weight="bold")
    plt.ylabel("Fraction of Samples within Tolerance", fontsize=13, weight="bold")
    plt.title(f"Regression Error Characteristic (REC) Curve - {model_name}\n(Unseen MOF Holdout)",
              fontsize=13, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{model_name}_REC_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_rec_curve(y_test, y_pred_test)

# --- Step 15: Comprehensive Excel File Export ---
output_path = "/users/hemanthsunkara/downloads/XGBoost_Comprehensive_Results.xlsx"

actuals_test_df = pd.DataFrame({
    "MOF_ID": groups_test.values,
    "Actual_Adsorption_Capacity_mg_g": y_test.values,
    "Predicted_Adsorption_Capacity_mg_g": y_pred_test
})

actuals_train_df = pd.DataFrame({
    "MOF_ID": groups_train.values,
    "Actual_Adsorption_Capacity_mg_g": y_train.values,
    "Predicted_Adsorption_Capacity_mg_g": y_pred_train
})

shap_xgb_df = pd.DataFrame(shap_values_xgb, columns=X.columns)

metrics_df = pd.DataFrame({
    "Metric": ["R²", "RMSE", "MAPE (%)", "sMAPE (%)", "MSLE", "RMSLE", "NRMSE"],
    "Blind Test Set (unseen MOFs)": [
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

split_info_df = pd.DataFrame({
    "Item": [
        "Total rows", "Total unique MOFs", "Train rows", "Train unique MOFs",
        "Blind Test rows", "Blind Test unique MOFs", "Outer Split Method",
        "Inner CV Method", "Locked CV Anchors", "Best Hyperparameters"
    ],
    "Value": [
        len(df), n_unique_mofs, len(X_train), groups_train.nunique(),
        len(X_test), groups_test.nunique(),
        "Kennard-Stone Space-Filling (Standardized Group Centroids)",
        "Anchor-Preserved Group Cross-Validation (k=5)",
        str(anchor_mofs),
        str(xgb_random.best_params_)
    ]
})

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    actuals_test_df.to_excel(writer, sheet_name="Actual_vs_Predicted_Test", index=False)
    actuals_train_df.to_excel(writer, sheet_name="Actual_vs_Predicted_Train", index=False)
    metrics_df.to_excel(writer, sheet_name="Summary_Metrics", index=False)
    shap_xgb_df.to_excel(writer, sheet_name="SHAP_Values", index=False)
    split_info_df.to_excel(writer, sheet_name="Split_Info", index=False)

print("\n" + "=" * 60)
print(f"All comprehensive data sheets saved successfully to: '{output_path}'")
print("  Exported Sheets:")
print("   1. Actual_vs_Predicted_Test")
print("   2. Actual_vs_Predicted_Train")
print("   3. Summary_Metrics")
print("   4. SHAP_Values")
print("   5. Split_Info")
print("=" * 60)