from datetime import datetime
from sqlalchemy import select
from src.data.database.models import ApartmentSaleListing, Location, Features


class ListingsRepository:
    def __init__(self, db):
        self.db = db

    def get(self, listing_id: int):
        with self.db.session() as session:
            return session.get(ApartmentSaleListing, listing_id)

    def list_all_ids(self):
        with self.db.session() as session:
            rows = session.execute(
                select(ApartmentSaleListing.id).order_by(ApartmentSaleListing.id.asc())
            ).all()
            return [r[0] for r in rows]

    def list_ids_by_period(self, date):
        with self.db.session() as session:
            rows = session.execute(
                select(ApartmentSaleListing.id)
                .where(ApartmentSaleListing.creation_date >= date)
                .order_by(ApartmentSaleListing.creation_date.asc())
            )
            return [r[0] for r in rows]

    def list_joined(self, city: str, since: datetime | None = None):
        with self.db.session() as session:
            query = (
                select(ApartmentSaleListing, Location, Features)
                .join(Location, ApartmentSaleListing.location_id == Location.id, isouter=True)
                .join(Features, ApartmentSaleListing.id == Features.listing_id, isouter=True)
                .where(Location.city == city)
            )
            if since is not None:
                query = query.where(ApartmentSaleListing.creation_date >= since).order_by(
                    ApartmentSaleListing.creation_date.desc(),
                    ApartmentSaleListing.creation_time.desc(),
                )
            return session.execute(query).all()
