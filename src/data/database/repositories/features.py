from src.data.database.models import Features


class FeaturesRepository:
    def __init__(self, db):
        self.db = db

    def get_by_listing(self, listing_id: int):
        with self.db.session() as session:
            return session.get(Features, listing_id)
