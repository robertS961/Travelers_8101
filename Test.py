df = pd.read_csv("C:\\Users\\zach\\Downloads\\Testing_TriGuard.csv")

# 如果你只想保留 zip3，而不要原来的 zip_code：
df = df.drop(columns=["zip_code"])

df["claim_date"] = pd.to_datetime(df["claim_date"])
df["claim_year"] = df["claim_date"].dt.year

# 2. 计算车龄 vehicle_age = claim_year - vehicle_made_year
df["vehicle_age"] = df["claim_year"] - df["vehicle_made_year"]
df["driver_age"] = df["claim_year"] - df["year_of_born"]
df["driver_age_clipped"] = df["driver_age"].clip(lower=16, upper=90)
df["driver_ability"] = df["driver_age_clipped"] - df["age_of_DL"]










df["past_num_of_claims_cap"] = df["past_num_of_claims"].clip(upper=5)
'''
df["has_past_claims"] = (df["past_num_of_claims"] > 0).astype(int)
df["is_high_freq_claim"] = (df["past_num_of_claims"] >= 3).astype(int)
df = df.drop(columns=["past_num_of_claims"])
'''





# 责任分箱：便于捕捉非线性
df["liab_bucket"] = pd.cut(
    df["liab_prct"],
    bins=[-1, 0, 20, 50, 80, 100],
    labels=["0", "0-20", "20-50", "50-80", "80-100"]
)

# 是否完全无责 / 重大责任
df["is_no_fault"] = (df["liab_prct"] == 0).astype(int)
df["is_major_fault"] = (df["liab_prct"] >= 50).astype(int)


# 是否有警察报告 & 证人
df["has_police_and_witness"] = (
    (df["policy_report_filed_ind"] == 1) & (df["witness_present_ind"] == 1)
).astype(int)

# 没有证人也没有警察报告（最难追偿的一类）
df["no_police_no_witness"] = (
    (df["policy_report_filed_ind"] == 0) & (df["witness_present_ind"] == 0)
).astype(int)

# 高责任但没警察报告
df["high_liab_no_report"] = (
    (df["liab_prct"] >= 50) & (df["policy_report_filed_ind"] == 0)
).astype(int)

# 多车事故 + 有警察报告
df["multi_vehicle_and_police"] = (
    df["accident_type"].isin(["multi_vehicle_unclear", "multi_vehicle_clear"])
    & (df["policy_report_filed_ind"] == 1)
).astype(int)

# 赔付/车价比（非常关键）
df["payout_ratio"] = df["claim_est_payout"] / (df["vehicle_price"] + 1)

# 是否在网络内修理厂
df["in_network_bodyshop_ind"] = (df["in_network_bodyshop"].astype(str).str.lower() == "yes").astype(int)

# 事故地点 × 是否网络内修理
df["site_x_bodyshop"] = df["accident_site"].astype(str) + "_" + df["in_network_bodyshop"].astype(str)

# 分箱
df["past_claims_bucket"] = pd.cut(
    df["past_num_of_claims"],
    bins=[-1, 0, 1, 3, 10],
    labels=["0", "1", "2-3", "4+"]
)

df["has_past_claims"] = (df["past_num_of_claims"] > 0).astype(int)
df["is_high_freq_claim"] = (df["past_num_of_claims"] >= 3).astype(int)



# 安全评级低
df["is_low_safety_rating"] = (df["safety_rating"] < df["safety_rating"].median()).astype(int)

# 近年搬家 + 租房
df["move_and_rent"] = (
    (df["address_change_ind"] == 1) & (df["living_status"] == "Rent")
).astype(int)

# 责任 × 赔付比
df["liab_x_payout_ratio"] = df["liab_prct"] * df["payout_ratio"]

# 责任 × 是否有警察报告
df["liab_x_police"] = df["liab_prct"] * df["policy_report_filed_ind"]

# 责任 × 是否有证人
df["liab_x_witness"] = df["liab_prct"] * df["witness_present_ind"].replace({'Y': 1, 'N': 0})

# 多次理赔 × 高责任
df["multi_claims_high_liab"] = (
    (df["past_num_of_claims"] >= 2) & (df["liab_prct"] >= 50)
).astype(int)


import numpy as np

long_tail_candidates = [
    "claim_est_payout",
    "vehicle_price",
    "annual_income",
    "vehicle_mileage",
    "past_num_of_claims"
]
import numpy as np

for col in ["claim_est_payout", "vehicle_price", "annual_income", "vehicle_mileage"]:
    df[col] = np.log1p(df[col])





# 3. 删掉原始的 claim_date 列（如果你只想保留年份）
df = df.drop(columns=["claim_date"])
df = df.drop(columns=["claim_year"])
df = df.drop(columns=["vehicle_made_year"])
df = df.drop(columns=["year_of_born"])
df = df.drop(columns=["driver_age"])
df = df.drop(columns=["age_of_DL"])
df = df.drop(columns=["claim_day_of_week"])

X_test = df.drop(columns=["subrogation", "claim_number"])




# 目标变量
y = df["subrogation"].astype(int)

# 特征（去掉 subrogation）
X = df.drop(columns=["subrogation"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("Unknown")
print(X.shape)


from sklearn.model_selection import train_test_split

# ⬆️ 可调比例，例如 train:val:test = 3:1:1
train_ratio = 4
val_ratio = 1


total = train_ratio + val_ratio

val_size = val_ratio / (train_ratio + val_ratio)  # 第二次划分要按剩余部分计算



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=val_size, random_state=42, stratify=y
)

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)


# 在训练集和验证集上都转成 category
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")
    X_test[c] = X_test[c].astype("category")



















import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import numpy as np
from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool

# 类别不平衡权重
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print("scale_pos_weight:", scale_pos_weight)

# CatBoost 需要类别特征的 column indices
cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_features_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_features_idx)

def objective_cat(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "iterations": trial.suggest_int("iterations", 300, 1500),
    }

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_state=42,
        verbose=False,
        scale_pos_weight=scale_pos_weight,
        task_type="CPU",   # 你有 GPU 的话可以改成 "GPU"
        **params
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
        verbose=False
    )

    y_proba = model.predict_proba(val_pool)[:, 1]

    thresholds = np.linspace(0.2, 0.8, 61)
    best_f1 = 0.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1


study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)

print("CatBoost Best F1 (val):", study_cat.best_value)
print("CatBoost Best params:", study_cat.best_params)

best_params_cat = study_cat.best_params.copy()

def find_best_threshold(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0.0
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print("Best threshold on val:", best_t, " Val F1:", best_f1)
    return best_t, best_f1


final_cat = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Logloss",
    random_state=42,
    verbose=False,
    scale_pos_weight=scale_pos_weight,
    task_type="CPU",
    **best_params_cat
)

final_cat.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=100,
    verbose=False
)




best_t_cat, best_f1_val_cat = find_best_threshold(final_cat, X_val, y_val)

y_test_proba_cat = final_cat.predict_proba(test_pool)[:, 1]
y_test_pred_cat = (y_test_proba_cat >= best_t_cat).astype(int)
test_f1_cat = f1_score(y_test, y_test_pred_cat)
print("CatBoost Test F1:", test_f1_cat)












import pandas as pd

# 假设 data 是包含 liab_prct 列的 DataFrame
# 1. 获取常用的描述性统计（包含 25%, 50%, 75% 分位数）
print(df['liab_prct'].describe())

# 2. 计算特定的分位数 (例如：95% 分位数，用于查看高负债人群)
q95 = data['liab_prct'].quantile(0.95)
print(f"95% 分位数是: {q95}")

# 3. 计算多个分位数
quantiles = data['liab_prct'].quantile([0.1, 0.5, 0.9])
print(quantiles)