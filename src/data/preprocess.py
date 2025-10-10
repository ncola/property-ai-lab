import pandas as pd
from datetime import date
from sklearn.impute import KNNImputer
import numpy as np

def basic_transformations_depending_on_database(df):
    out = df.copy()

    out = out.replace(["NONE","None", "none", "NA", "<na>", "null", "NULL", ""], pd.NA)

    converters = {
        "price_per_m":  lambda s: pd.to_numeric(s, errors="coerce").astype(float),
        "area":         lambda s: pd.to_numeric(s, errors="coerce").astype(float),
        "building_build_year": lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
        "creation_date":        lambda s: pd.to_datetime(s, errors="coerce"),
    }

    for col, func in converters.items():
        if col in out.columns:
            out[col] = func(out[col])

    return out


def price_per_m2_outliers(df):
    low, high = df['price_per_m'].quantile([0.01, 0.99])
    df_clean = df[(df['price_per_m'] >= low) & (df['price_per_m'] <= high)].copy()
    return df_clean


def price_per_m2_nan(df):
    out = df.copy()

    if "price_per_m" in out.columns:
        out["price_per_m"] = pd.to_numeric(out["price_per_m"], errors="coerce")
    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if "area" in out.columns:
        out["area"] = pd.to_numeric(out["area"], errors="coerce")

    if {"price", "area"}.issubset(out.columns):
        mask = out["price_per_m"].isna() & out["price"].notna() & out["area"].notna() & (out["area"] > 0)
        out.loc[mask, "price_per_m"] = out.loc[mask, "price"] / out.loc[mask, "area"]

    out = out.dropna(subset=["price_per_m"]).copy()

    out["price_per_m"] = out["price_per_m"].astype(float)

    return out
     

def clean_building_build_year(df):
    out = df.copy()

    year_col = "building_build_year"
    status_col = "construction_status"

    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out[status_col] = out[status_col].astype("string").str.strip().str.lower()

    to_completion_mask = out[year_col].isna() & (out[status_col] == "to_completion")
    out.loc[to_completion_mask, year_col] = -1

    knn_mask = out[year_col].isna()
    if knn_mask.any():
        feature_cols = ["construction_status", "market", "price_per_m", "district", "building_type", "area"]

        cat_cols = ["construction_status", "market", "district", "building_type"]
        num_cols = ["price_per_m", "area"]

        for col in num_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        features = pd.get_dummies(out[feature_cols], columns=cat_cols, dummy_na=True)

        mat = pd.concat([features, out[[year_col]]], axis=1)

        mat = mat.astype(float)

        has_known_years = mat[year_col].notna().sum() > 0
        if has_known_years and features.shape[1] > 0:
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            imputed = imputer.fit_transform(mat.values)
            imputed_df = pd.DataFrame(imputed, columns=mat.columns, index=mat.index)
            imputed_years = imputed_df[year_col]

            fill_mask = knn_mask
            out.loc[fill_mask, year_col] = np.round(imputed_years.loc[fill_mask]).astype("float")
        else:
            if out[year_col].notna().any():
                median_year = float(out[year_col].dropna().median())
            else:
                median_year = np.nan  

            out.loc[knn_mask, year_col] = median_year

        still_nan = out[year_col].isna()
        if still_nan.any():
            group_cols = ["district", "building_type"]
            grp_med = (
                out.groupby(group_cols, dropna=False)[year_col]
                .transform(lambda s: s.fillna(s.median()))
            )
            out[year_col] = out[year_col].fillna(grp_med)

            if out[year_col].isna().any() and out[year_col].notna().any():
                out[year_col] = out[year_col].fillna(out[year_col].median())

    return out


def _compute_age(year):
    today = date.today()
    if pd.isna(year):          
        return pd.NA
    if year == -1:           
        return -1
    return today.year - int(year)

def create_building_age(df):
    df["building_build_year"] = pd.to_numeric(df["building_build_year"], errors="coerce")
    df["building_age"] = df["building_build_year"].apply(_compute_age).astype("Int64")
    return df


def to_categorical(df):
    df_cat = df.copy()
    obj_cols = df_cat.select_dtypes(include=["object", "string"]).columns

    cat_cols = []
    for col in obj_cols:
        df_cat[col] = df_cat[col].astype("category")
        cat_cols.append(col)

    return df_cat, cat_cols

def to_float64(df):
    df_float = df.copy()
    num_cols = df_float.select_dtypes(include=["number"]).columns.tolist()
    df_float[num_cols] = df_float[num_cols].astype("float64")

    return df_float, num_cols
