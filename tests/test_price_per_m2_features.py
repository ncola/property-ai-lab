import pandas as pd
import pytest

from src.data.price_per_m2_features import add_price_per_m2_features


def test_adds_price_per_m2_model_features():
    df = pd.DataFrame({
        "building_age": [5, 10],
        "floor_num": ["0", "10+"],
        "rooms_num": [2, 3],
        "area": [50.0, 75.0],
        "building_floors_num": [4, 12],
    })

    result = add_price_per_m2_features(df)

    assert result.loc[0, "building_age"] == 5.0
    assert result.loc[0, "is_ground"] == 1
    assert result.loc[0, "is_low"] == 1
    assert result.loc[1, "is_10plus"] == 1
    assert result.loc[0, "rooms_per_area"] == pytest.approx(0.04)
    assert result.loc[1, "rooms_per_area"] == pytest.approx(0.04)
