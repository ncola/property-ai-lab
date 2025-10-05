from typing import Optional
from datetime import date
from decimal import Decimal
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, BigInteger, Text, Date, Boolean, Numeric, String
from db_setup import Base

class ApartmentSaleListing(Base):
    __tablename__ = "apartments_sale_listings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)

    otodom_listing_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    market: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    advert_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    creation_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    creation_time: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pushed_up_at: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    exclusive_offer: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    creation_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    area: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    price: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    updated_price: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    price_per_m: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    updated_price_per_m: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    location_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    street: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rent_amount: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rooms_num: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    floor_num: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    heating: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ownership: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    proper_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    construction_status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    energy_certificate: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    building_build_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    building_floors_num: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    building_material: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    building_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    windows_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_plan_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    video_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    view3d_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    walkaround_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    owner_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    owner_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    agency_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    agency_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    offer_link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    closing_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    development_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    development_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class Features(Base):
    __tablename__ = "features"

    listing_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)

    internet: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    cable_television: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    phone: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    roller_shutters: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    anti_burglary_door: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    entryphone: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    monitoring: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    alarm: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    closed_area: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    furniture: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    washing_machine: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    dishwasher: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    fridge: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    stove: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    oven: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    tv: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    balcony: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    usable_room: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    garage: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    basement: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    garden: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    terrace: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    lift: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    two_storey: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    separate_kitchen: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    air_conditioning: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)

    voivodeship: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    city: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    district: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
