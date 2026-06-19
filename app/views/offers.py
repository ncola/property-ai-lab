import streamlit as st
import pandas as pd

from app.utils import fmt_money, cut_description, is_deal
from src.data.database.db_setup import Database
from src.data.database.service import DataService
from src.serving.inference import predict


st.title("🏠 Property Price Predictor")
st.markdown("Najświezsze oferty mieszkań na sprzedaz w Katowicach z otodom")


@st.cache_data(show_spinner=True)
def load_offers_by_period(period):
    db = Database()
    return DataService(db).load_data_by_period(period)


@st.cache_data(show_spinner="Liczę predykcje...")
def predict_cached(df: pd.DataFrame) -> pd.Series:
    return pd.Series(predict(df), index=df.index)


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

try:
    predicted = predict_cached(df_all)
except Exception as e:
    st.error(f"Błąd predykcji modelu: {e}")
    st.stop()

df_all = df_all.copy()
df_all["predicted_price"] = predicted.astype(int)
df_all["predicted_price_per_m2"] = (df_all["predicted_price"] / df_all["area"]).astype(int)
diff, deals = is_deal(df_all)
df_all["diff"] = (diff * 100).round(2)
df_all["is_deal"] = deals


# =========================
# Filters & sorting sidebar
# =========================
MARKET_LABELS = {"primary": "pierwotny", "secondary": "wtórny"}

FEATURE_OPTIONS = {
    "balcony": "balkon",
    "terrace": "taras",
    "garden": "ogród",
    "garage": "garaż",
    "basement": "piwnica",
    "lift": "winda",
    "air_conditioning": "klimatyzacja",
    "furniture": "umeblowane",
    "separate_kitchen": "oddzielna kuchnia",
    "closed_area": "ogrodzony teren",
    "monitoring": "monitoring",
}

SORT_OPTIONS = {
    "Najnowsze": ("creation_date", False),
    "Najlepsze potencjalne okazje (%)": ("diff", False),
    "Cena: rosnąco": ("price", True),
    "Cena: malejąco": ("price", False),
    "Cena za m²: rosnąco": ("price_per_m", True),
    "Cena za m²: malejąco": ("price_per_m", False),
    "Powierzchnia: rosnąco": ("area", True),
    "Powierzchnia: malejąco": ("area", False),
    "Rok budowy: najnowsze": ("building_build_year", False),
}


def _range_slider(label, series, step=None, fmt=None, key=None):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    lo, hi = s.min(), s.max()
    if lo == hi:
        st.caption(f"{label}: {lo}")
        return (lo, hi)
    lo, hi = (int(lo), int(hi)) if step is None or isinstance(step, int) else (float(lo), float(hi))
    kwargs = {"step": step} if step is not None else {}
    if fmt is not None:
        kwargs["format"] = fmt
    if key is not None:
        kwargs["key"] = key
    return st.slider(label, lo, hi, (lo, hi), **kwargs)


FILTER_KEYS = [
    "f_districts", "f_markets", "f_deal", "f_price", "f_ppm", "f_area",
    "f_rooms", "f_year", "f_floors", "f_features", "f_desc", "f_sort",
]

with st.sidebar:
    header_col, btn_col = st.columns([2, 1], vertical_alignment="center")
    with header_col:
        st.markdown("### 🔎 Filtry")
    with btn_col:
        if st.button("Wyczyść", width="stretch"):
            for k in FILTER_KEYS:
                st.session_state.pop(k, None)
            st.rerun()

    districts = sorted(df_all["district"].dropna().astype(str).unique().tolist())
    sel_districts = st.multiselect("Dzielnica", districts, key="f_districts")

    markets_present = [m for m in df_all["market"].dropna().unique().tolist() if m in MARKET_LABELS or True]
    sel_markets = st.multiselect(
        "Rynek",
        options=markets_present,
        format_func=lambda x: MARKET_LABELS.get(x, x),
        key="f_markets",
    )

    deal_filter = st.radio(
        "Potencjalne okazje",
        options=["wszystkie", "tylko potencjalne okazje", "bez potencjalnych okazji"],
        horizontal=True,
        key="f_deal",
    )

    sel_price = _range_slider("Cena (zł)", df_all["price"], step=10000, key="f_price")
    sel_ppm = _range_slider("Cena za m² (zł)", df_all["price_per_m"], step=100, key="f_ppm")
    sel_area = _range_slider("Powierzchnia (m²)", df_all["area"], step=1, key="f_area")

    rooms_vals = sorted(pd.to_numeric(df_all["rooms_num"], errors="coerce").dropna().astype(int).unique().tolist())
    sel_rooms = st.multiselect("Liczba pokoi", rooms_vals, key="f_rooms")

    sel_year = _range_slider("Rok budowy", df_all["building_build_year"], step=1, key="f_year")

    floor_vals = sorted(df_all["floor_num"].dropna().astype(str).unique().tolist())
    sel_floors = st.multiselect("Piętro", floor_vals, key="f_floors")

    available_features = [c for c in FEATURE_OPTIONS if c in df_all.columns]
    sel_features = st.multiselect(
        "Wymagane udogodnienia",
        options=available_features,
        format_func=lambda c: FEATURE_OPTIONS[c],
        key="f_features",
    )

    sel_desc_query = st.text_input(
        "Szukaj w opisie",
        placeholder="np. osiedle Tuwima",
        key="f_desc",
    )

    st.markdown("### ↕️ Sortowanie")
    sort_key = st.selectbox("Sortuj według", list(SORT_OPTIONS.keys()), key="f_sort")


# =========================
# Apply filters
# =========================
df_view = df_all.copy()

if sel_districts:
    df_view = df_view[df_view["district"].astype(str).isin(sel_districts)]
if sel_markets:
    df_view = df_view[df_view["market"].isin(sel_markets)]
if sel_price is not None:
    df_view = df_view[df_view["price"].between(sel_price[0], sel_price[1])]
if sel_ppm is not None:
    df_view = df_view[df_view["price_per_m"].between(sel_ppm[0], sel_ppm[1])]
if sel_area is not None:
    df_view = df_view[df_view["area"].between(sel_area[0], sel_area[1])]
if sel_rooms:
    df_view = df_view[pd.to_numeric(df_view["rooms_num"], errors="coerce").isin(sel_rooms)]
if sel_year is not None:
    df_view = df_view[df_view["building_build_year"].between(sel_year[0], sel_year[1])]
if sel_floors:
    df_view = df_view[df_view["floor_num"].astype(str).isin(sel_floors)]
for feat in sel_features:
    df_view = df_view[df_view[feat] == True]

if sel_desc_query and sel_desc_query.strip():
    df_view = df_view[
        df_view["description_text"].fillna("").astype(str).str.contains(
            sel_desc_query.strip(), case=False, regex=False
        )
    ]

if deal_filter == "tylko potencjalne okazje":
    df_view = df_view[df_view["is_deal"]]
elif deal_filter == "bez potencjalnych okazji":
    df_view = df_view[~df_view["is_deal"]]

# =========================
# Sort
# =========================
sort_col, ascending = SORT_OPTIONS[sort_key]
if sort_col in df_view.columns:
    df_view = df_view.sort_values(sort_col, ascending=ascending, na_position="last")

# =========================
# Summary + count slider
# =========================
total_in_period = len(df_all)
total_found = len(df_view)
deals_found = int(df_view["is_deal"].sum()) if total_found else 0

st.markdown(
    f"### Oferty — znaleziono **{total_found}** / {total_in_period} "
    f"(w tym potencjalnych okazji: **{deals_found}**)"
)

if total_found == 0:
    st.warning("Brak ofert spełniających wybrane filtry.")
    st.stop()

if total_found == 1:
    n_cards = 1
else:
    n_cards = st.slider(
        "Liczba ofert do pokazania",
        min_value=1,
        max_value=total_found,
        value=min(10, total_found),
        step=1,
    )
df_view = df_view.head(int(n_cards))


# =========================
# Cards
# =========================
icons = {
    "creation_date": "🕓",
    "district": "📍",
    "area": "📐",
    "floor_num": "🏢",
    "market": "🏬",
    "building_build_year": "🏗️",
}

info_order = ["creation_date", "district", "area", "floor_num", "market", "building_build_year"]

for i, row in df_view.iterrows():
    card = st.container(border=True)

    with card:
        left, right = st.columns([2, 1], vertical_alignment="top")

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
                        st.button("zwiń ▲", key=f"btn_{i}", width="stretch",
                                  on_click=lambda k=desc_key: st.session_state.update({k: False}))
            else:
                st.markdown(f"Opis:<br>{fmt_desc(cut_description(description, 400))}", unsafe_allow_html=True)
                _, btn_col, _ = st.columns([1, 1, 1])
                with btn_col:
                    st.button("pokaż pełny opis ▼", key=f"btn_{i}", width="stretch",
                              on_click=lambda k=desc_key: st.session_state.update({k: True}))
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
                    val = f"{'pierwotny' if val == 'primary' else 'wtórny'}"
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
                st.dataframe(pd.DataFrame(row).T, width="stretch")

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
                st.markdown(f"##### Potencjalna okazja")
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
