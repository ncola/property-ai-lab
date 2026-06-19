import pandas as pd
from datetime import date


def clean_raw_data(df):
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


def ensure_price_per_m2(df):
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


def _compute_age(year):
    if pd.isna(year):
        return pd.NA
    return date.today().year - int(year)

def create_building_age(df):
    df["building_build_year"] = pd.to_numeric(df["building_build_year"], errors="coerce")
    df["building_age"] = df["building_build_year"].apply(_compute_age).astype("Int64")
    return df


def _decode_if_bytes(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8")
    if isinstance(v, memoryview):
        return bytes(v).decode("utf-8")
    return v


def to_categorical(df):
    df_cat = df.copy()
    obj_cols = df_cat.select_dtypes(include=["object", "string"]).columns

    cat_cols = []
    for col in obj_cols:
        df_cat[col] = df_cat[col].map(_decode_if_bytes).astype("category")
        cat_cols.append(col)

    return df_cat, cat_cols

def to_float64(df):
    df_float = df.copy()
    num_cols = df_float.select_dtypes(include=["number"]).columns.tolist()
    df_float[num_cols] = df_float[num_cols].astype("float64")

    return df_float, num_cols
