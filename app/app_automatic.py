import streamlit as st
import pandas as pd
from utils import fmt_money, cut_description, calculate_price, is_deal
from src.data.database.db_setup import Database
from src.data.database.service import DataService
from src.serving.inference import features_inference_pipeline, predict
from src.utils.mlflow_utils import set_tracking


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

try:
    set_tracking()
except Exception as e:
    st.error("Couldn't set mlflow tracking uri, check config: {e}")
    st.stop()

# choosing period
st.markdown("### Ustawienia")
period_label = st.segmented_control(
    "Okres",
    options=["ostatnie 3 dni", "ostatni tydzień", "ostatni miesiąc"],
    selection_mode="single",
    default="ostatnie 3 dni"
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

# choosing top rows
n_cards = st.slider("Liczba ofert", min_value=1, max_value=len(df_all)-1, value=min(10, len(df_all)-1), step=1)
df_all = df_all.head(int(n_cards))


# add model inference
X = features_inference_pipeline(df_all)
predicted = predict(X)

df_all["predicted_price_per_m2"] = predicted
df_all["predicted_price"] = calculate_price(df_all)
diff, deals = is_deal(df_all)
df_all["diff"] = round(diff*100, 2)
df_all["is_deal"] = deals



st.markdown("### Oferty")
# view
icons = {
    "creation_date": "🕓",
    "district": "📍",
    "area": "📐",
    "floor_num": "🏢",
    "market": "🏬",
    "building_build_year": "🏗️",
}

info_order = ["creation_date", "district", "area", "floor_num", "market", "building_build_year"]

for i, row in df_all.head(int(n_cards)).iterrows():
    card = st.container(border=True)

    with card:
        left, right = st.columns([2,1], vertical_alignment="top")

        with left:
            title = row.get("title") or f"Oferta #{row.get('otodom_listing_id', i)}"
            st.markdown(f"##### {title}")
            description = row.get("description_text")
            st.markdown(f"Opis: {cut_description(description, 400)}")
            link = row.get("offer_link")
            st.markdown(f"🔗 [Link do oferty]({link})")

            info_display = []
            for key in info_order:
                val = row.get(key, None)

                if key == "creation_date" and pd.notna(val):
                    val = f"{val}"
                elif key == "area" and pd.notna(val):
                    val = f"{val} m²"
                elif key == "floor_num" and pd.notna(val):
                    val = f"Piętro {val}"
                elif key == "market" and pd.notna(val):
                    val =f"{'pierwotny' if val=='primary' else 'wtórny'}"
                elif key == "building_build_year" and pd.notna(val):
                    val = int(val)
                elif pd.isna(val) or val is None or str(val).strip() == "":
                    val = "brak informacji"

                info_display.append((icons.get(key), val))

            cols = st.columns(len(info_display))
            for col, (icon, val) in zip(cols, info_display):
                col.markdown(
                    f"""
                    <div style='text-align:center; font-size:15px; line-height:1.9'>
                        {icon}<br>{val}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
            with st.expander("Podgląd danych"):
                st.dataframe(pd.DataFrame(row).T, use_container_width=True)



        with right:
            col_true, col_pred, col_deal = st.columns(3)

            with col_true:
                st.markdown(f"##### Rzeczywiste")
                true_ppm = row.get("price_per_m")
                true_price = row.get("price")
                st.metric("cena", fmt_money(true_price))
                st.metric("cena za m2", fmt_money(true_ppm))

            with col_pred:
                st.markdown(f"##### Predykcja")
                pred_ppm = row.get("predicted_price_per_m2")
                pred_total = row.get("predicted_price")
                st.metric("cena", fmt_money(pred_total))
                st.metric("cena za m2", fmt_money(pred_ppm))

            with col_deal:
                st.markdown(f"##### Okazja")
                deal = row.get("is_deal") 
                diff = row.get("diff")
                st.metric("róznica", f"{diff}%")
                color = "#6ECD71" if deal else "#F57A71"  
                text = "tak" if deal else "nie"

                st.markdown(
                    f"""
                    potencjalna okazja <br>
                    <div style="
                        background-color: {color};
                        color: white;
                        padding: 0.4rem 0;
                        border-radius: 8px;
                        text-align: center;
                        font-weight: bold;
                        font-size: 1.1rem;
                    ">
                        {text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )





                    

