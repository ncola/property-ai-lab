import pandas as pd

import src.data.preprocess as pre


def add_price_per_m2_features(df):
    out = df.copy()
    out = _add_building_age_features(out)
    out = _add_floor_features(out)
    out = _add_room_area_features(out)
    return out


def _add_building_age_features(df):
    if "building_age" not in df.columns and "building_build_year" in df.columns:
        df = pre.create_building_age(df)
    elif "building_age" not in df.columns:
        df["building_age"] = pd.NA
    df["building_age"] = pd.to_numeric(df["building_age"], errors="coerce").astype("float64")
    return df


def _add_floor_features(df):
    if "floor_num" not in df.columns:
        df["floor_num"] = pd.NA

    df["floor_num"] = df["floor_num"].replace("10+", 10)
    df["floor_num"] = pd.to_numeric(df["floor_num"], errors="coerce")
    df["is_ground"] = df["floor_num"].eq(0).fillna(False).astype("int64")
    df["is_low"] = df["floor_num"].le(2).fillna(False).astype("int64")
    df["is_10plus"] = df["floor_num"].ge(10).fillna(False).astype("int64")
    return df


def _add_room_area_features(df):
    for col in ("rooms_num", "area", "building_floors_num"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rooms_per_area"] = df["rooms_num"] / df["area"]
    return df
