import pandas as pd
import streamlit as st

from app.utils import cut_description, fmt_money, is_deal
from src.data.database.db_setup import Database
from src.data.database.service import DataService
from src.serving.inference import predict


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
button[kind="secondary"] {
  background: none !important;
  border: none !important;
  color: #888 !important;
  font-size: 0.8rem !important;
  box-shadow: none !important;
}
button[kind="secondary"]:hover {
  color: #333 !important;
  text-decoration: underline;
  background: none !important;
}
div[data-testid="stButton"] {
  margin-top: -1rem !important;
}
</style>
""", unsafe_allow_html=True)

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
predicted = predict(df_all)

df_all["predicted_price"] = predicted
df_all["predicted_price_per_m2"] = (predicted / df_all["area"]).astype(int)
diff, deals = is_deal(df_all)
df_all["diff"] = round(diff*100, 2)
df_all["is_deal"] = deals


st.markdown("### Oferty")
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
            description = row.get("description_text") or ""
            desc_key = f"desc_expanded_{i}"
            if desc_key not in st.session_state:
                st.session_state[desc_key] = False
            def fmt_desc(text):
                return "<br>".join(text.splitlines())

            if st.session_state[desc_key] or len(description) <= 400:
                st.markdown(f"Opis:<br>{fmt_desc(description)}", unsafe_allow_html=True)
                if len(description) > 400:
                    _, btn_col, _ = st.columns([1, 1, 1])
                    with btn_col:
                        st.button("zwiń ▲", key=f"btn_{i}", use_container_width=True, on_click=lambda k=desc_key: st.session_state.update({k: False}))
            else:
                st.markdown(f"Opis:<br>{fmt_desc(cut_description(description, 400))}", unsafe_allow_html=True)
                _, btn_col, _ = st.columns([1, 1, 1])
                with btn_col:
                    st.button("pokaż pełny opis ▼", key=f"btn_{i}", use_container_width=True, on_click=lambda k=desc_key: st.session_state.update({k: True}))
            link = row.get("offer_link")
            st.markdown(f"🔗 [Link do oferty]({link})")

            info_display = []
            for key in info_order:
                val = row.get(key, None)

                if key == "creation_date" and pd.notna(val):
                    time = row.get("creation_time")
                    val = f"{val}\n{str(time)[:5]}" if pd.notna(time) and str(time).strip() else f"{val}"
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
