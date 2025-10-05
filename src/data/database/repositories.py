from db_setup import Database
from models import ApartmentSaleListing, Location, Features
from sqlalchemy import select

class ListingsRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, listing_id: int):
        with self.db.session() as session:
            return session.get(ApartmentSaleListing, listing_id)
        
    def list_all_ids(self) -> list[int]:
        with self.db.session() as session:
            rows = session.execute(
                select(ApartmentSaleListing.id).order_by(ApartmentSaleListing.id.asc())
            ).all()
            return [r[0] for r in rows]


class FeaturesRepository:
    def __init__(self, db: Database):
        self.db = db

    def get_by_listing(self, listing_id: int):
        with self.db.session() as session:
            return session.get(Features, listing_id)


class LocationsRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, location_id: int):
        with self.db.session() as s:
            return s.get(Location, location_id)

