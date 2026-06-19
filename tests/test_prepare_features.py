import pandas as pd
import numpy as np
import pytest
from src.data.prepare_features import features_train_pipeline


INCLUDE = ["area", "rooms_num", "building_age"]
TARGET = "price_per_m"


def make_df(**overrides):
    base = {
        "price_per_m":          [5000.0, 6000.0, 7000.0, 8000.0, 5500.0],
        "area":                 [50.0,   60.0,   70.0,   80.0,   55.0],
        "rooms_num":            [2.0,    3.0,    4.0,    2.0,    3.0],
        "building_build_year":  [2000,   2005,   2010,   2015,   2020],
        "price":                [250000, 360000, 490000, 640000, 302500],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestFeaturesTrainPipeline:
    def test_returns_x_y_and_cat_cols(self):
        X, y, cat_cols = features_train_pipeline(make_df(), TARGET, INCLUDE)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(cat_cols, list)

    def test_target_not_in_x(self):
        X, _, _ = features_train_pipeline(make_df(), TARGET, INCLUDE)
        assert TARGET not in X.columns

    def test_x_has_only_included_columns(self):
        X, _, _ = features_train_pipeline(make_df(), TARGET, INCLUDE)
        assert set(X.columns) == set(INCLUDE)

    def test_x_and_y_same_length(self):
        X, y, _ = features_train_pipeline(make_df(), TARGET, INCLUDE)
        assert len(X) == len(y)

    def test_drops_rows_where_target_is_null(self):
        # tworzymy duży df żeby outlier filter nie usunął normalnych wierszy
        n = 100
        df = pd.DataFrame({
            "price_per_m":         [5000.0] * n,
            "area":                [50.0] * n,
            "rooms_num":           [2.0] * n,
            "building_build_year": [2000] * n,
            "price":               [250000] * n,
        })
        df.loc[0, "price_per_m"] = None
        X, y, _ = features_train_pipeline(df, TARGET, INCLUDE)
        assert len(y) == n - 1

    def test_drops_all_nan_columns(self):
        df = make_df()
        df["floor_num"] = np.nan
        include_with_empty = INCLUDE + ["floor_num"]
        X, _, _ = features_train_pipeline(df, TARGET, include_with_empty)
        assert "floor_num" not in X.columns

    def test_x_columns_are_float64(self):
        X, _, _ = features_train_pipeline(make_df(), TARGET, INCLUDE)
        for col in X.select_dtypes(include="number").columns:
            assert X[col].dtype == np.float64
