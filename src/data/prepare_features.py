import pandas as pd
import src.data.preprocess as pre


def features_train_pipeline(df_raw, target_col, include):
    df = pre.basic_transformations_depending_on_database(df_raw.copy())
    df = pre.price_per_m2_outliers(df)
    #df = pre.clean_building_build_year(df)
    df = pre.create_building_age(df)

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])

    X = X[include]

    empty_cols = [c for c in X.columns if X[c].isna().all()]
    if empty_cols:
        X = X.drop(columns=empty_cols)

    X, cat_cols = pre.to_categorical(X)
    X, _ = pre.to_float64(X)

    return X, y, cat_cols

