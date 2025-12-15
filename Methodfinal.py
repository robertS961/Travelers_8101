import numpy as np
import pandas as pd
import gc
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool

# Assume df_train and test_raw already exist
# Assume feature_engineering function is already defined, for example:
# def feature_engineering(df, is_train=True):
#     # ... your feature engineering code ...
#     return df

# ==========================================
# 1. Data Preparation
# ==========================================

# Assume df_train already exists
# Please replace with your actual data loading:
# df_train = pd.read_csv("train.csv")
# test_raw = pd.read_csv("test.csv")

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

# Simple imputation for numerical columns
X_tree[num_cols] = X_tree[num_cols].fillna(X_tree[num_cols].median())

# Categorical indices required by CatBoost
cat_features_idx = [X_tree.columns.get_loc(c) for c in cat_cols]

# --- Preparation B: Linear Model Data (Base Logistic Regression) ---
# Logistic Regression requires One-Hot Encoding and Standardization
X_linear = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)

# Impute numerical missing values
for col in num_cols:
    X_linear[col] = X_linear[col].fillna(X_linear[col].median())

# Standardization
scaler = StandardScaler()
X_linear_scaled = pd.DataFrame(
    scaler.fit_transform(X_linear),
    columns=X_linear.columns,
    index=X_linear.index
)

print(f"Tree Shape: {X_tree.shape}, Linear Shape: {X_linear_scaled.shape}")

# ==========================================
# 2. Optuna Automatic Tuning (Get Best Hyperparameters)
# ==========================================

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
N_TRIALS = 50 # Optuna search count, adjust based on computing resources

# Use only the first fold data for fast tuning
initial_train_idx, initial_val_idx = next(skf.split(X_tree, y_full))
X_tr_opt, X_val_opt = X_tree.iloc[initial_train_idx], X_tree.iloc[initial_val_idx]
y_tr_opt, y_val_opt = y_full.iloc[initial_train_idx], y_full.iloc[initial_val_idx]

# CatBoost Pool for tuning
train_pool_opt = Pool(X_tr_opt, y_tr_opt, cat_features=cat_features_idx)
val_pool_opt = Pool(X_val_opt, y_val_opt, cat_features=cat_features_idx)

print("\n--- Starting Optuna Tuning for Base Models (LGBM & CatBoost) ---")

# --- A. Optuna Objective Function - LightGBM ---
def objective_lgbm(trial):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_tr_opt, y_tr_opt,
        eval_set=[(X_val_opt, y_val_opt)],
        eval_metric='binary_logloss',
        callbacks=[early_stopping(50), log_evaluation(0)]
    )
    return model.best_score_['valid_0']['binary_logloss']

# --- B. Optuna Objective Function - CatBoost ---
def objective_catboost(trial):
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'Logloss', 'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_seed': 42, 'verbose': False, 'allow_writing_files': False,
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool_opt, eval_set=val_pool_opt, early_stopping_rounds=50, verbose=False)
    return model.get_best_score()['validation']['Logloss']


# Execute tuning
study_lgbm = optuna.create_study(direction='minimize', study_name='LGBM_Tuning')
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=True)
best_lgbm_params = study_lgbm.best_params
print("\n[LightGBM] Best Hyperparameters Found:")
print(best_lgbm_params)

study_catboost = optuna.create_study(direction='minimize', study_name='CatBoost_Tuning')
study_catboost.optimize(objective_catboost, n_trials=N_TRIALS, show_progress_bar=True)
best_cat_params = study_catboost.best_params
print("\n[CatBoost] Best Hyperparameters Found:")
print(best_cat_params)
print("-------------------------------------------------------------------")


# ==========================================
# 3. K-Fold OOF Stacking (Use Tuned Parameters)
# ==========================================

# LGBM Final Parameters
lgbm_final_params = {
    'objective': 'binary', 'random_state': 42, 'n_estimators': 1000, 'n_jobs': -1, 'verbose': -1,
    **best_lgbm_params
}

# CatBoost Final Parameters
catboost_final_params = {
    'loss_function': 'Logloss', 'eval_metric': 'Logloss',
    'iterations': 1000, 'random_seed': 42, 'verbose': False, 'allow_writing_files': False,
    **best_cat_params
}

oof_lgb = np.zeros(len(X_tree))
oof_cat = np.zeros(len(X_tree))
oof_lr = np.zeros(len(X_tree))

models_lgb = []
models_cat = []
models_lr = []

print(f"\nStarting {N_FOLDS}-Fold Stacking with Tuned Parameters...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tree, y_full)):
    print(f"\n=== Fold {fold + 1} / {N_FOLDS} ===")

    # Split data
    X_tr_tree, X_val_tree = X_tree.iloc[train_idx], X_tree.iloc[val_idx]
    X_tr_lin, X_val_lin = X_linear_scaled.iloc[train_idx], X_linear_scaled.iloc[val_idx]
    y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

    # --- Model 1: LightGBM (Use Tuned Parameters) ---
    lgb = LGBMClassifier(**lgbm_final_params)
    lgb.fit(
        X_tr_tree, y_tr,
        eval_set=[(X_val_tree, y_val)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(50), log_evaluation(0)]
    )
    oof_lgb[val_idx] = lgb.predict_proba(X_val_tree)[:, 1]
    models_lgb.append(lgb)

    # --- Model 2: CatBoost (Use Tuned Parameters) ---
    train_pool = Pool(X_tr_tree, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_val_tree, y_val, cat_features=cat_features_idx)
    cat = CatBoostClassifier(**catboost_final_params)
    cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
    oof_cat[val_idx] = cat.predict_proba(val_pool)[:, 1]
    models_cat.append(cat)

    # --- Model 3: Logistic Regression (Base) ---
    lr_base = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    lr_base.fit(X_tr_lin, y_tr)
    oof_lr[val_idx] = lr_base.predict_proba(X_val_lin)[:, 1]
    models_lr.append(lr_base)

# ==========================================
# 4. Meta Model Training (Based on OOF Results)
# ==========================================

# Concatenate OOF predictions from three models as features for Meta Model
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


# ==========================================
# 5. Test Prediction
# ==========================================

# Assume test_raw is already loaded
test_claim_id = test_raw["claim_number"].copy()

# -----------------
# 5.1 Test Data Processing (Must strictly align with training set)
# -----------------

# Note: Ensure feature_engineering function returns correct DataFrame structure on test set.

# -- A. Tree Data --
# ... (Previous code remains unchanged, e.g., X_test_tree = feature_engineering(test_raw, is_train=False) )

# Assume test_raw is already loaded
# test_raw = pd.read_csv(...)
test_claim_id = test_raw["claim_number"].copy()

# -----------------
# 4.1 Test Data Processing (Must strictly align with training set)
# -----------------

# -- A. Tree Data --
X_test_tree = feature_engineering(test_raw, is_train=False)  # Assume you have this function
# Fill missing columns
for col in cat_cols:
    X_test_tree[col] = X_test_tree[col].astype(str).astype("category")
    if "Unknown" not in X_test_tree[col].cat.categories:
        X_test_tree[col] = X_test_tree[col].cat.add_categories(["Unknown"])
    X_test_tree[col] = X_test_tree[col].fillna("Unknown")

X_test_tree[num_cols] = X_test_tree[num_cols].fillna(X_tree[num_cols].median())  # Use Train median
X_test_tree = X_test_tree[X_tree.columns]  # Ensure column order consistency

# -- B. Linear Data --
X_test_lin_raw = feature_engineering(test_raw, is_train=False)
X_test_lin = pd.get_dummies(X_test_lin_raw, columns=cat_cols, dummy_na=True)

# Align columns: Fill 0 if present in Train but not Test, drop if present in Test but not Train
X_test_lin = X_test_lin.reindex(columns=X_linear.columns, fill_value=0)

# Impute values (Use Train data)
for col in num_cols:
    X_test_lin[col] = X_test_lin[col].fillna(X_linear[col].median())

# Standardization (Use scaler fitted on Train)
X_test_lin_scaled = pd.DataFrame(
    scaler.transform(X_test_lin),
    columns=X_test_lin.columns
)

# -----------------
# 5.2 Prediction (Average of 5 folds)
# -----------------

test_pred_lgb = np.zeros(len(X_test_tree))
test_pred_cat = np.zeros(len(X_test_tree))
test_pred_lr = np.zeros(len(X_test_tree))

# Iterate through saved 5 models
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

# Generate Submission File
submission = pd.DataFrame({
    "claim_number": test_claim_id.values,
    "subrogation": final_preds
})

print("\n--- Submission Head ---")
print(submission.head())

# ==========================================
# 6. Local F1 Score Evaluation (Assume df_train contains true labels)
# ==========================================

# 1. Prepare True Label Data
# Note: We use OOF results for evaluation here because this is the training data for the Meta Model
# The accuracy of OOF results best reflects the model's generalization ability.
y_true_oof = y_full.values
y_pred_oof = (meta_probs >= best_t).astype(int)

# 2. Calculate OOF F1 Score
# Note: This score was printed above, repeated here to clearly show evaluation steps
score_oof = f1_score(y_true_oof, y_pred_oof)

print(f"\nFinal Stacking Model OOF F1 Score: {score_oof:.5f}")