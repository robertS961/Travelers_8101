import pandas as pd
import numpy as np


def feature_engineering(raw_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Perform feature engineering on raw TriGuard data.
    When is_train=True, keep 'subrogation' and drop rows where 'subrogation' is NaN.
    When is_train=False, leave 'subrogation' untouched (usually this column doesn't exist in test set).
    """
    df = raw_df.copy()

    # ====== 1. Basic Cleaning ======
    drop_basic = ["claim_number", "zip_code"]
    df = df.drop(columns=[c for c in drop_basic if c in df.columns])

    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["claim_year"] = df["claim_date"].dt.year

    # ====== 2. Feature Engineering ======
    # Use the section organized previously; simplified here for demonstration,
    # please replace with the full version in practice.

    df["vehicle_age"] = df["claim_year"] - df["vehicle_made_year"]
    df["driver_age"] = df["claim_year"] - df["year_of_born"]
    df["driver_age_clipped"] = df["driver_age"].clip(lower=16, upper=90)
    df["driver_ability"] = df["driver_age_clipped"] - df["age_of_DL"]

    df["past_num_of_claims_cap"] = df["past_num_of_claims"].clip(upper=5)

    df["liab_bucket"] = pd.cut(
        df["liab_prct"],
        bins=[-1, 0, 20, 50, 80, 100],
        labels=["0", "0-20", "20-50", "50-80", "80-100"]
    )
    df["liab_gt_50"] = (df["liab_prct"] > 50).astype(int)
    df["liab_gt_80"] = (df["liab_prct"] > 80).astype(int)
    df["liab_ratio"] = df["liab_prct"] / 100

    df["is_no_fault"] = (df["liab_prct"] == 0).astype(int)
    df["is_major_fault"] = (df["liab_prct"] >= 50).astype(int)

    df["witness_present_ind"] = df["witness_present_ind"].map({'Y': 1, 'N': 0}).astype("Int64")

    df["has_police_and_witness"] = (
            (df["policy_report_filed_ind"] == 1) & (df["witness_present_ind"] == 1)
    ).astype(int)
    df["no_police_no_witness"] = (
            (df["policy_report_filed_ind"] == 0) & (df["witness_present_ind"] == 0)
    ).astype(int)
    df["high_liab_no_report"] = (
            (df["liab_prct"] >= 50) & (df["policy_report_filed_ind"] == 0)
    ).astype(int)
    df["multi_vehicle_and_police"] = (
            df["accident_type"].isin(["multi_vehicle_unclear", "multi_vehicle_clear"])
            & (df["policy_report_filed_ind"] == 1)
    ).astype(int)

    df["is_clean_case"] = (
            (df["witness_present_ind"] == 1) & (df["policy_report_filed_ind"] == 1)
    ).astype(int)
    df["is_messy_case"] = (
            (df["witness_present_ind"] == 0) & (df["policy_report_filed_ind"] == 0)
    ).astype(int)

    df["payout_ratio"] = df["claim_est_payout"] / (df["vehicle_price"] + 1)
    df["payout_ratio_bin"] = pd.qcut(df["payout_ratio"], q=10, duplicates="drop")
    df["liab_x_payout"] = df["liab_prct"] * df["payout_ratio"]
    df["liab_x_payout_ratio"] = df["liab_prct"] * df["payout_ratio"]

    df["in_network_bodyshop_ind"] = (
            df["in_network_bodyshop"].astype(str).str.lower() == "yes"
    ).astype(int)
    df["site_x_bodyshop"] = (
            df["accident_site"].astype(str) + "_" + df["in_network_bodyshop"].astype(str)
    )

    df["past_claims_bucket"] = pd.cut(
        df["past_num_of_claims"],
        bins=[-1, 0, 1, 3, 10],
        labels=["0", "1", "2-3", "4+"]
    )
    df["has_past_claims"] = (df["past_num_of_claims"] > 0).astype(int)
    df["is_high_freq_claim"] = (df["past_num_of_claims"] >= 3).astype(int)

    df["is_low_safety_rating"] = (df["safety_rating"] < df["safety_rating"].median()).astype(int)
    df["move_and_rent"] = (
            (df["address_change_ind"] == 1) & (df["living_status"] == "Rent")
    ).astype(int)

    long_tail_cols = [
        "claim_est_payout",
        "vehicle_price",
        "annual_income",
        "vehicle_mileage",
        "past_num_of_claims",
        "vehicle_weight",
    ]
    for col in long_tail_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    df["years_norm"] = (df["driver_years"] - df["driver_years"].mean()) / (
                df["driver_years"].max() - df["driver_years"].min())
    df["safety_norm"] = (df["safety_rating"] - df["safety_rating"].mean()) / (
                df["safety_rating"].max() - df["driver_years"].min())
    df["claims_norm"] = (df["past_num_of_claims_cap"] - df["past_num_of_claims_cap"].mean()) / (
                df["past_num_of_claims_cap"].max() - df["past_num_of_claims_cap"].min())

    df["driver_risk_latent"] = (
            0.4 * (1 - df["years_norm"]) +
            0.4 * (1 - df["safety_norm"]) +
            0.2 * df["claims_norm"]
    )

    df['vehicle_depr_value'] = (
            df['vehicle_price'] *
            (1 - annual_rate * df['vehicle_age']) *
            (1 - mile_rate * df['vehicle_mileage'])
    )

    drop_cols_2 = [
        "claim_date",
        "claim_year",
        "vehicle_made_year",
        "year_of_born",
        "driver_age",
        "age_of_DL",
        "claim_day_of_week",
        "gender",
        "vehicle_color",
        "channel",
        "annual_income",
        "has_past_claims",
    ]
    df = df.drop(columns=[c for c in drop_cols_2 if c in df.columns])

    # Only the training set needs to drop rows with missing labels; test set does not have 'subrogation'
    if is_train and "subrogation" in df.columns:
        df = df[~df["subrogation"].isna()].copy()

    return df


df = pd.read_csv("C:\\Users\\zach\\Downloads\\Training_TriGuard.csv")

import pandas as pd
from sklearn.model_selection import train_test_split

# test_size=0.2 means the test set takes up 20% (i.e., a 4:1 split)
random_state = 42
train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=42)

# Print shapes to verify
print(f"Raw data: {df.shape}")
print(f"Training set (train_raw): {train_raw.shape}")
print(f"Test set (test_raw): {test_raw.shape}")

'''### When we use test data
train_raw = pd.read_csv("C:\\Users\\zach\\Downloads\\Training_TriGuard.csv")
test_raw = pd.read_csv("C:\\Users\\zach\\Downloads\\Testing_TriGuard.csv")'''

df_train = feature_engineering(train_raw, is_train=True)
df_test_proc = feature_engineering(test_raw, is_train=False)