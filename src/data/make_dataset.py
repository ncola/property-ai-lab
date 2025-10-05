import pandas as pd
import numpy as np

from database.repositories import Database
from database.models import ApartmentSaleListing, Location, Features
from database.repositories import LocationsRepository, ListingsRepository, FeaturesRepository


class TrainingDataService:
    def __init__(self, db: Database):
        self.db = db
        self.listings = ListingsRepository(db)
        self.locations = LocationsRepository(db)
        self.features = FeaturesRepository(db)

    def object_to_dict(self, obj):
        """ORM object -> dict"""
        if obj is None:
            return {}
        return {k: v for k, v in vars(obj).items() if not k.startswith("_sa_")}

    def load_dataframe_all(self):
        all_listing_ids = self.listings.list_all_ids()
        if not all_listing_ids:
            return pd.DataFrame()

        rows = []
        for listing_id in all_listing_ids:
            listing = self.listings.get(int(listing_id))
            if listing is None:
                continue

            location_obj = None
            if getattr(listing, "location_id", None) is not None:
                try:
                    location_obj = self.locations.get(int(listing.location_id))
                except Exception:
                    location_obj = self.locations.get(listing.location_id)

            features_obj = self.features.get_by_listing(int(listing_id))

            row = {
                **self.object_to_dict(listing),
                **self.object_to_dict(location_obj),
                **self.object_to_dict(features_obj),
                }
            rows.append(row)

        df = pd.DataFrame(rows)

        df = df[df["city"] == "katowice"]

        return df