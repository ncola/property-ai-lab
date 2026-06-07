import pytest
import pandas as pd


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "price_per_m": [5000.0, 6000.0, None, 8000.0],
        "price":       [250000, 360000, 300000, 400000],
        "area":        [50.0, 60.0, 75.0, 50.0],
    })
