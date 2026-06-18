import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import (
    ensure_price_per_m2,
    _compute_age,
    create_building_age,
    clean_raw_data,
    to_categorical,
    to_float64,
)


class TestEnsurePricePerM2:
    def test_fills_missing_from_price_and_area(self, sample_df):
        result = ensure_price_per_m2(sample_df)
        assert result.loc[2, "price_per_m"] == pytest.approx(4000.0)

    def test_drops_rows_without_price_and_area(self):
        df = pd.DataFrame({
            "price_per_m": [None],
            "price":       [None],
            "area":        [50.0],
        })
        result = ensure_price_per_m2(df)
        assert len(result) == 0

    def test_no_nulls_in_result(self, sample_df):
        result = ensure_price_per_m2(sample_df)
        assert result["price_per_m"].isna().sum() == 0


class TestComputeAge:
    def test_normal_year(self):
        age = _compute_age(2000)
        assert age == 2026 - 2000

    def test_nan_returns_na(self):
        assert pd.isna(_compute_age(float("nan")))


class TestCreateBuildingAge:
    def test_adds_building_age_column(self):
        df = pd.DataFrame({"building_build_year": [2000, 2010]})
        result = create_building_age(df)
        assert "building_age" in result.columns

    def test_age_is_correct(self):
        df = pd.DataFrame({"building_build_year": [2000]})
        result = create_building_age(df)
        assert result.loc[0, "building_age"] == 2026 - 2000

    def test_nan_year_gives_nan_age(self):
        df = pd.DataFrame({"building_build_year": [None]})
        result = create_building_age(df)
        assert pd.isna(result.loc[0, "building_age"])


class TestBasicTransformations:
    def test_replaces_none_strings_with_na(self):
        df = pd.DataFrame({"price_per_m": ["NONE", "None", "null", "NULL", "NA", ""]})
        result = clean_raw_data(df)
        assert result["price_per_m"].isna().all()

    def test_converts_price_per_m_to_float(self):
        df = pd.DataFrame({"price_per_m": ["5000", "6000"]})
        result = clean_raw_data(df)
        assert result["price_per_m"].dtype == float

    def test_converts_building_build_year_to_int(self):
        df = pd.DataFrame({"building_build_year": ["2000", "2010"]})
        result = clean_raw_data(df)
        assert pd.api.types.is_integer_dtype(result["building_build_year"])

    def test_converts_creation_date_to_datetime(self):
        df = pd.DataFrame({"creation_date": ["2024-01-01", "2024-06-15"]})
        result = clean_raw_data(df)
        assert pd.api.types.is_datetime64_any_dtype(result["creation_date"])

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"price_per_m": ["5000"]})
        _ = clean_raw_data(df)
        assert df["price_per_m"].dtype == object


class TestToCategorical:
    def test_converts_object_cols_to_category(self):
        df = pd.DataFrame({"city": ["Warsaw", "Krakow"], "price": [5000.0, 6000.0]})
        result, cat_cols = to_categorical(df)
        assert result["city"].dtype.name == "category"
        assert "city" in cat_cols

    def test_numeric_cols_unchanged(self):
        df = pd.DataFrame({"city": ["Warsaw"], "price": [5000.0]})
        result, _ = to_categorical(df)
        assert result["price"].dtype == float


class TestToFloat64:
    def test_converts_numeric_cols_to_float64(self):
        df = pd.DataFrame({"area": pd.array([50, 60], dtype="Int64"), "city": ["Warsaw", "Krakow"]})
        result, num_cols = to_float64(df)
        assert result["area"].dtype == np.float64
        assert "area" in num_cols

    def test_non_numeric_cols_unchanged(self):
        df = pd.DataFrame({"area": [50.0], "city": ["Warsaw"]})
        result, _ = to_float64(df)
        assert result["city"].dtype == object
