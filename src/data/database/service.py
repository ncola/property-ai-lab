from datetime import datetime, timedelta

import pandas as pd

from src.data.database.repositories.listings import ListingsRepository
from src.data.database.repositories.locations import LocationsRepository
from src.data.database.repositories.features import FeaturesRepository


_PERIOD_DAYS = {"last_3d": 3, "last_week": 7, "last_month": 30}


def _to_dict(obj):
    if obj is None:
        return {}
    return {k: v for k, v in vars(obj).items() if not k.startswith("_sa_")}


class DataService:
    def __init__(self, db):
        self.db = db
        self.listings = ListingsRepository(db)
        self.locations = LocationsRepository(db)
        self.features = FeaturesRepository(db)

    def load_dataframe(self, city: str = "katowice", since: datetime | None = None) -> pd.DataFrame:
        rows = self.listings.list_joined(city, since)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(
            {**_to_dict(listing), **_to_dict(location), **_to_dict(features)}
            for listing, location, features in rows
        )

    def load_dataframe_all(self) -> pd.DataFrame:
        return self.load_dataframe()

    def load_data_by_period(self, period: str = "last_week") -> pd.DataFrame:
        if period not in _PERIOD_DAYS:
            raise ValueError(f"Not valid period: {period}")
        since = datetime.now() - timedelta(days=_PERIOD_DAYS[period])
        return self.load_dataframe(since=since)
