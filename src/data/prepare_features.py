import pandas as pd
import src.data.preprocess as pre


def prepare_features(df_raw, target_col,include):
    df = pre.basic_transformations_depending_on_database(df_raw.copy())
    df = pre.price_per_m2_outliers(df)
    df = pre.clean_building_build_year(df)
    df = pre.create_building_age(df)

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    feature_cols = [c for c in include if c in df.columns]

    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        elif df[c].dtype == "O":
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass

    categorical_fixed = ["market", "building_material", "construction_status", "district", "floor_num"]
    for c in categorical_fixed:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("__NA__")

    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    cat_cols = [c for c in categorical_fixed if c in feature_cols]

    return df, feature_cols, cat_cols