import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from src.data.database.db_setup import Database
from src.data.database.service import DataService


st.set_page_config(page_title="Property AI Lab 🏠", layout="wide")
st.title("🏠 Property Price Predictor")
st.markdown("Najświezsze oferty mieszkań na sprzedaz w Katowicach z otodom")

@st.cache_data(show_spinner=True)
def load_offers_by_period(period):
    db = Database()
    return DataService(db).load_data_by_period(period)

st.markdown("""
<style>
.block-container {
  max-width: 99vw !important; 
  padding-left: 2rem;
  padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)


#try:
#    mlflow.set_tracking_uri("http://127.0.0.1:5002")  
#    MODEL_URI = "runs:/acb8b2b5f9a243bfa8c0371bcbeda0af/model"
#    booster = mlflow.xgboost.load_model(MODEL_URI)
#except Exception as e:
#    st.error(f"Nie udało się załadować modelu dla {MODEL_URI}: {e}")
#    st.stop()

# choosing period
period_label = st.segmented_control(
    "Okres:",
    options=["ostatnie 3 dni", "ostatni tydzień", "ostatni miesiąc"],
    selection_mode="single",
    default="ostatni tydzień"
)
period_map = {
    "ostatnie 3 dni": "last_3d",
    "ostatni tydzień": "last_week",
    "ostatni miesiąc": "last_month",
}
period = period_map[period_label]


# get data from database
try:
    df_all = load_offers_by_period(period)
except Exception as e:
    st.error(f"Błąd ładowania danych z bazy: {e}")
    st.stop()

if df_all.empty:
    st.warning("Brak danych w bazie.")
    st.stop()

# filtering columns
available_cols = list(df_all.columns)
default_cols = [c for c in ["otodom_listing_id", "title", "offer_link", "description_text", "creation_date","market", "building_build_year", "area", "rooms_num", "building_floors_num", "floor_num", "building_material", "construction_status", "district", 'balcony', 'separate_kitchen', 'air_conditioning','roller_shutters', 'dishwasher', 'garage','anti_burglary_door', 'basement', 'entryphone','garden', 'internet', 'monitoring', 'terrace', 'alarm', 'lift', 'closed_area'] if c in df_all.columns]
with st.sidebar:
    st.markdown("### ⚙️ Widoczne kolumny")
    visible_cols = st.multiselect(
        "Wybierz kolumny do wyświetlenia:",
        options=available_cols,
        default=default_cols
    )

# add model inference
df_all["predicted price per m2"] = 5000
df_all["predicted price"] = 500000

st.write("#### Oferty z wybranego okresu")
st.dataframe(df_all[visible_cols].reset_index(drop=True), width='stretch')
