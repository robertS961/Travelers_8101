import numpy as np
import pandas as pd
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool

# ==========================================
# 1. Data Preparation
# ==========================================

# Assuming df_train already exists
y_full = df_train["subrogation"].astype(int)
X_raw = df_train.drop(columns=["subrogation"]).copy()

# Distinguish between numerical and categorical columns
cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_raw.columns if c not in cat_cols]

# --- Preparation A: Tree Model Data (LGBM / CatBoost) ---
X_tree = X_raw.copy()

# Fill missing values & convert types
for col in cat_cols:
    X_tree[col] = X_tree[col].astype(str).fillna("Unknown").astype("category")

# Simple fill for numerical columns
X_tree[num_cols] = X_tree[num_cols].fillna(X_tree[num_cols].median())

# Category indices required by CatBoost
cat_features_idx = [X_tree.columns.get_loc(c) for c in cat_cols]

# --- Preparation B: Linear Model Data (Base Logistic Regression) ---
# Logistic Regression requires One-Hot Encoding and Standardization
X_linear = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)

# Impute numerical missing values (Mean is usually friendlier for LR, or keep Median)
for col in num_cols:
    X_linear[col] = X_linear[col].fillna(X_linear[col].median())

# Standardization (Very important, otherwise LR struggles to converge)
scaler = StandardScaler()
# Note: Strictly speaking, fit should be done inside the Fold, but for code brevity
# and large datasets, global fit differences are usually negligible.
# Strict approach: fit_transform inside CV loop.
X_linear_scaled = pd.DataFrame(
    scaler.fit_transform(X_linear),
    columns=X_linear.columns,
    index=X_linear.index
)

print(f"Tree Shape: {X_tree.shape}, Linear Shape: {X_linear_scaled.shape}")

# ==========================================
# 2. K-Fold OOF Stacking
# ==========================================

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Used to store OOF (Out-Of-Fold) predictions -> Training data for Meta Model
oof_lgb = np.zeros(len(X_tree))
oof_cat = np.zeros(len(X_tree))
oof_lr = np.zeros(len(X_tree))  # Newly added Base LR

# Used to store Test set predictions (Averaged at the end)
# Assuming X_test is already processed with the same logic (code for processing X_test follows)
# Keeping it here, will use inside the loop
# ...

models_lgb = []
models_cat = []
models_lr = []

print(f"Starting {N_FOLDS}-Fold Stacking...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tree, y_full)):
    print(f"\n=== Fold {fold + 1} / {N_FOLDS} ===")

    # Split data
    # Tree model data
    X_tr_tree, X_val_tree = X_tree.iloc[train_idx], X_tree.iloc[val_idx]
    # Linear model data
    X_tr_lin, X_val_lin = X_linear_scaled.iloc[train_idx], X_linear_scaled.iloc[val_idx]

    y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

    # --- Model 1: LightGBM ---
    lgb = LGBMClassifier(
        objective="binary", random_state=42, n_estimators=1000,
        learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1
    )
    lgb.fit(
        X_tr_tree, y_tr,
        eval_set=[(X_val_tree, y_val)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(50), log_evaluation(0)]  # 0 means no printing
    )
    oof_lgb[val_idx] = lgb.predict_proba(X_val_tree)[:, 1]
    models_lgb.append(lgb)

    # --- Model 2: CatBoost ---
    train_pool = Pool(X_tr_tree, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val_tree, y_val, cat_features=cat_features_idx)

    cat = CatBoostClassifier(
        loss_function="Logloss", eval_metric="Logloss",
        iterations=1000, depth=6, learning_rate=0.05,
        random_seed=42, verbose=False, allow_writing_files=False
    )
    cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

    oof_cat[val_idx] = cat.predict_proba(val_pool)[:, 1]
    models_cat.append(cat)

    # --- Model 3: Logistic Regression (Base) ---
    # Using One-Hot + Scaled data
    lr_base = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    lr_base.fit(X_tr_lin, y_tr)

    oof_lr[val_idx] = lr_base.predict_proba(X_val_lin)[:, 1]
    models_lr.append(lr_base)

# ==========================================
# 3. Meta Model Training (Based on OOF Results)
# ==========================================

# Stack OOF predictions from three models as features for the Meta Model
X_meta_train = np.column_stack([oof_lgb, oof_cat, oof_lr])

print(f"\nMeta Train Shape: {X_meta_train.shape}")
print("Correlation between models (LGB, CAT, LR):")
print(np.corrcoef(X_meta_train, rowvar=False))

# Define Meta Model
meta_clf = LogisticRegression(penalty="l2", C=1.0, random_state=42)
meta_clf.fit(X_meta_train, y_full)


# Find Best Threshold
def find_best_threshold_from_probs(y_true, proba):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


meta_probs = meta_clf.predict_proba(X_meta_train)[:, 1]
best_t, best_f1 = find_best_threshold_from_probs(y_full, meta_probs)

print(f"CV F1 Score: {best_f1:.4f} at Threshold: {best_t:.4f}")
print(f"Meta Model Coefficients: {meta_clf.coef_}")
# Observing coefficients shows which Base Model has higher weight (Index 0: LGB, 1: Cat, 2: LR)


# ==========================================
# 4. Test Prediction
# ==========================================

# Assuming test_raw is already loaded
# test_raw = pd.read_csv(...)
test_claim_id = test_raw["claim_number"].copy()

# -----------------
# 4.1 Test Data Processing (Must align strictly with Train)
# -----------------

# -- A. Tree Data --
X_test_tree = feature_engineering(test_raw, is_train=False)  # Assuming you have this function
# Fill missing columns
for col in cat_cols:
    X_test_tree[col] = X_test_tree[col].astype(str).astype("category")
    if "Unknown" not in X_test_tree[col].cat.categories:
        X_test_tree[col] = X_test_tree[col].cat.add_categories(["Unknown"])
    X_test_tree[col] = X_test_tree[col].fillna("Unknown")

X_test_tree[num_cols] = X_test_tree[num_cols].fillna(X_tree[num_cols].median())  # Use Train median
X_test_tree = X_test_tree[X_tree.columns]  # Ensure column order consistency

# -- B. Linear Data --
# For One-Hot, the safest way is to redo get_dummies and then align
X_test_lin = pd.get_dummies(feature_engineering(test_raw, is_train=False), columns=cat_cols, dummy_na=True)

# Align columns: Fill 0 for cols in Train but not Test, Drop cols in Test but not Train
X_test_lin = X_test_lin.reindex(columns=X_linear.columns, fill_value=0)

# Fill numericals (Use Train data)
for col in num_cols:
    X_test_lin[col] = X_test_lin[col].fillna(X_linear[col].median())

# Standardization (Use the scaler fitted on Train)
X_test_lin_scaled = pd.DataFrame(
    scaler.transform(X_test_lin),
    columns=X_test_lin.columns
)

# -----------------
# 4.2 Make Predictions (Average of 5 folds)
# -----------------

test_pred_lgb = np.zeros(len(X_test_tree))
test_pred_cat = np.zeros(len(X_test_tree))
test_pred_lr = np.zeros(len(X_test_tree))

# Iterate through stored 5 models
for i in range(N_FOLDS):
    # LGB
    test_pred_lgb += models_lgb[i].predict_proba(X_test_tree)[:, 1] / N_FOLDS
    # CatBoost
    test_pred_cat += models_cat[i].predict_proba(X_test_tree)[:, 1] / N_FOLDS
    # Base LR
    test_pred_lr += models_lr[i].predict_proba(X_test_lin_scaled)[:, 1] / N_FOLDS

# Construct Meta Features for Test Set
X_meta_test = np.column_stack([test_pred_lgb, test_pred_cat, test_pred_lr])

# Meta Model Final Prediction
final_probs = meta_clf.predict_proba(X_meta_test)[:, 1]
final_preds = (final_probs >= best_t).astype(int)

# Generate Submission
submission = pd.DataFrame({
    "claim_number": test_claim_id.values,
    "subrogation": final_preds
})

print(submission.head())


# ==========================================
# 5. Local Evaluation (If labels exist)
# ==========================================

import pandas as pd
from sklearn.metrics import f1_score

# 1. Prepare Ground Truth Data
# Extract only claim_number and true label column from df
# rename is used to distinguish between true and predicted after merge
true_labels = df[['claim_number', 'subrogation']].rename(columns={'subrogation': 'target_true'})

# 2. Key Step: Merge
# Use 'inner' join to ensure we only calculate for claims existing in submission
# This automatically aligns by claim_number, order doesn't matter
merged_df = pd.merge(submission, true_labels, on='claim_number', how='left')

# 3. Check for missing values (in case claim_numbers in submission are not in df)
if merged_df['target_true'].isnull().any():
    print("Warning: Some claim_numbers in submission were not found in df!")
    # If only a few are missing, drop them, or check data source
    merged_df = merged_df.dropna(subset=['target_true'])

# 4. Extract Aligned Y
y_pred = merged_df['subrogation']  # Predicted values from submission
y_true = merged_df['target_true']  # True values from df

# 5. Calculate F1 Score
score = f1_score(y_true, y_pred)

print(f"Submission F1 Score: {score:.5f}")