import pandas as pd
from datetime import datetime, timedelta
from src.data.database.repositories import LocationsRepository, ListingsRepository, FeaturesRepository

class DataService:
    def __init__(self, db):
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
        from src.data.database.models import ApartmentSaleListing, Location, Features
        from sqlalchemy import select, join

        with self.db.session() as session:
            query = select(
                ApartmentSaleListing,
                Location,
                Features
            ).join(
                Location, ApartmentSaleListing.location_id == Location.id, isouter=True
            ).join(
                Features, ApartmentSaleListing.id == Features.listing_id, isouter=True
            ).where(Location.city == "katowice")

            results = session.execute(query).all()

            if not results:
                return pd.DataFrame()

            rows = []
            for listing, location, features in results:
                row = {
                    **self.object_to_dict(listing),
                    **self.object_to_dict(location),
                    **self.object_to_dict(features),
                }
                rows.append(row)

            return pd.DataFrame(rows)

    def load_data_by_period(self, period="last_week"):
        from src.data.database.models import ApartmentSaleListing, Location, Features
        from sqlalchemy import select, join

        today = datetime.now()

        if period == "last_3d":
            date = today - timedelta(days=3)
        elif period == "last_week":
            date = today - timedelta(days=7)
        elif period == "last_month":
            date = today - timedelta(days=30)
        else:
            raise ValueError(f"Not valid period: {period}")

        with self.db.session() as session:
            query = select(
                ApartmentSaleListing,
                Location,
                Features
            ).join(
                Location, ApartmentSaleListing.location_id == Location.id, isouter=True
            ).join(
                Features, ApartmentSaleListing.id == Features.listing_id, isouter=True
            ).where(
                (Location.city == "katowice") &
                (ApartmentSaleListing.creation_date >= date)
            ).order_by(
                ApartmentSaleListing.creation_date.desc(),
                ApartmentSaleListing.creation_time.desc()
            )

            results = session.execute(query).all()

            if not results:
                return pd.DataFrame()

            rows = []
            for listing, location, features in results:
                row = {
                    **self.object_to_dict(listing),
                    **self.object_to_dict(location),
                    **self.object_to_dict(features),
                }
                rows.append(row)

            return pd.DataFrame(rows)
    


