import pandas as pd
import pytest
from src.data.preprocess import price_per_m2_nan, price_per_m2_outliers, _compute_age


class TestPricePerM2Nan:
    def test_fills_missing_from_price_and_area(self, sample_df):
        result = price_per_m2_nan(sample_df)
        # wiersz z indeksem 2 miał None w price_per_m: 300000 / 75 = 4000.0
        assert result.loc[2, "price_per_m"] == pytest.approx(4000.0)

    def test_drops_rows_without_price_and_area(self):
        df = pd.DataFrame({
            "price_per_m": [None],
            "price":       [None],
            "area":        [50.0],
        })
        result = price_per_m2_nan(df)
        assert len(result) == 0

    def test_no_nulls_in_result(self, sample_df):
        result = price_per_m2_nan(sample_df)
        assert result["price_per_m"].isna().sum() == 0


class TestPricePerM2Outliers:
    def test_removes_extreme_values(self):
        # 100 wierszy: 98 normalnych, 1 bardzo niski, 1 bardzo wysoki
        normal = [5000.0] * 98
        df = pd.DataFrame({"price_per_m": [1.0] + normal + [9_999_999.0]})
        result = price_per_m2_outliers(df)
        assert result["price_per_m"].max() < 9_999_999.0
        assert result["price_per_m"].min() > 1.0

    def test_keeps_majority_of_rows(self):
        normal = [5000.0] * 98
        df = pd.DataFrame({"price_per_m": [1.0] + normal + [9_999_999.0]})
        result = price_per_m2_outliers(df)
        assert len(result) >= 96


class TestComputeAge:
    def test_normal_year(self):
        age = _compute_age(2000)
        assert age == 2026 - 2000

    def test_to_completion_returns_minus_one(self):
        assert _compute_age(-1) == -1

    def test_nan_returns_na(self):
        assert pd.isna(_compute_age(float("nan")))
