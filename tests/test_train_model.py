import numpy as np
import pandas as pd
import pytest
import yaml

from src.pipelines.train_model import (
    TARGET_COL,
    build_price_per_m2_xy,
    prepare_price_per_m2_training_frame,
)


def make_training_df(n=20):
    rows = []
    for market in ("primary", "secondary"):
        for i in range(n):
            area = 50.0 + (i % 5)
            price_per_m = 10_000.0 if market == "primary" else 8_000.0
            rows.append({
                "market": market,
                "price": area * price_per_m,
                "price_per_m": price_per_m,
                "area": area,
                "building_build_year": 2010,
                "rooms_num": 2 + (i % 3),
                "building_floors_num": 5,
                "floor_num": "0" if i % 2 == 0 else "10+",
                "building_material": "brick",
                "construction_status": "ready_to_use",
                "district": "Śródmieście",
                "balcony": True,
                "separate_kitchen": False,
                "dishwasher": True,
                "garage": False,
                "anti_burglary_door": False,
                "basement": True,
                "entryphone": True,
                "garden": False,
                "monitoring": False,
                "terrace": False,
                "lift": True,
                "closed_area": False,
                "building_type": "block",
                "furniture": False,
                "usable_room": False,
                "advert_type": "agency",
            })
    return pd.DataFrame(rows)


def inference_features():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)["inference"]["feature_columns"]


def test_prepare_price_per_m2_training_frame_computes_log_target():
    df = prepare_price_per_m2_training_frame(make_training_df())

    assert TARGET_COL in df.columns
    assert set(df["market"]) == {"primary", "secondary"}
    assert df.loc[df["market"] == "primary", TARGET_COL].iloc[0] == pytest.approx(np.log(10_000))
    assert df.loc[df["market"] == "secondary", TARGET_COL].iloc[0] == pytest.approx(np.log(8_000))


def test_prepare_price_per_m2_training_frame_trims_outliers_per_market():
    df_raw = make_training_df(n=100)
    outliers = []
    for market in ("primary", "secondary"):
        base = df_raw[df_raw["market"] == market].iloc[0].copy()
        low = base.copy()
        low["price_per_m"] = 1.0
        low["price"] = low["area"] * low["price_per_m"]
        high = base.copy()
        high["price_per_m"] = 999_999.0
        high["price"] = high["area"] * high["price_per_m"]
        outliers.extend([low, high])

    df = prepare_price_per_m2_training_frame(pd.concat([df_raw, pd.DataFrame(outliers)], ignore_index=True))

    for market in ("primary", "secondary"):
        market_prices = df.loc[df["market"] == market, "price_per_m"]
        assert market_prices.min() > 1.0
        assert market_prices.max() < 999_999.0


def test_build_price_per_m2_xy_matches_inference_features_without_area():
    df = prepare_price_per_m2_training_frame(make_training_df())
    X, y, aux, cat_cols = build_price_per_m2_xy(
        df[df["market"] == "primary"],
        inference_features(),
    )

    assert list(X.columns) == inference_features()
    assert "area" not in X.columns
    assert len(X) == len(y) == len(aux)
    assert "district" in cat_cols
    assert str(X["district"].dtype) == "category"
