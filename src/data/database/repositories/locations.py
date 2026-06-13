from src.data.database.models import Location


class LocationsRepository:
    def __init__(self, db):
        self.db = db

    def get(self, location_id: int):
        with self.db.session() as session:
            return session.get(Location, location_id)
