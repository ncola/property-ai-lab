import numpy as np
import pandas as pd
import pytest

import src.serving.inference as inference


PRICE_PER_M2_FEATURES = [
    "building_age",
    "rooms_num",
    "building_floors_num",
    "floor_num",
    "building_material",
    "construction_status",
    "district",
    "balcony",
    "separate_kitchen",
    "dishwasher",
    "garage",
    "anti_burglary_door",
    "basement",
    "entryphone",
    "garden",
    "monitoring",
    "terrace",
    "lift",
    "closed_area",
    "is_ground",
    "is_low",
    "is_10plus",
    "rooms_per_area",
    "building_type",
    "furniture",
    "usable_room",
    "advert_type",
]


def price_per_m2_model_cfg():
    return {
        "inference": {
            "feature_columns": PRICE_PER_M2_FEATURES,
            "model_uris": {
                "primary": "primary-uri",
                "secondary": "secondary-uri",
            },
        }
    }


class DummyModel:
    def __init__(self, price_per_m2):
        self.price_per_m2 = price_per_m2
        self.last_X = None

    def predict(self, X):
        self.last_X = X.copy()
        return np.log(np.full(len(X), self.price_per_m2, dtype=float))


def make_input_df():
    return pd.DataFrame({
        "market": ["primary", "secondary"],
        "building_age": [0, 20],
        "area": [50.0, 60.0],
        "rooms_num": [2, 3],
        "building_floors_num": [4, 10],
        "floor_num": ["0", "10+"],
        "building_material": ["brick", "concrete_plate"],
        "construction_status": ["ready_to_use", "to_renovation"],
        "district": ["Śródmieście", "Koszutka"],
        "balcony": [True, False],
        "separate_kitchen": [False, True],
        "dishwasher": [True, False],
        "garage": [False, True],
        "anti_burglary_door": [False, False],
        "basement": [True, True],
        "entryphone": [True, True],
        "garden": [False, False],
        "monitoring": [False, True],
        "terrace": [False, False],
        "lift": [True, True],
        "closed_area": [False, False],
        "building_type": ["block", "tenement"],
        "furniture": [False, True],
        "usable_room": [False, True],
        "advert_type": ["developer_unit", "agency"],
    })


def test_predict_price_per_m2_model_uses_market_model_and_area_transform(monkeypatch):
    primary_model = DummyModel(price_per_m2=10_000)
    secondary_model = DummyModel(price_per_m2=8_000)
    models = {
        "primary-uri": primary_model,
        "secondary-uri": secondary_model,
    }

    monkeypatch.setattr(inference, "_load_config", price_per_m2_model_cfg)
    monkeypatch.setattr(inference, "_load_model", lambda uri: models[uri])

    predicted = inference.predict(make_input_df())

    assert predicted.tolist() == [500_000, 480_000]
    assert "area" not in primary_model.last_X.columns
    assert primary_model.last_X.iloc[0]["rooms_per_area"] == pytest.approx(0.04)
    assert secondary_model.last_X.iloc[0]["is_10plus"] == pytest.approx(1.0)
    assert str(primary_model.last_X["building_material"].dtype) == "category"
    assert primary_model.last_X["balcony"].dtype == bool


def test_predict_price_per_m2_model_rejects_missing_area(monkeypatch):
    df = make_input_df()
    df.loc[0, "area"] = None

    monkeypatch.setattr(inference, "_load_config", price_per_m2_model_cfg)

    with pytest.raises(ValueError, match="Area must be present"):
        inference.predict(df)
