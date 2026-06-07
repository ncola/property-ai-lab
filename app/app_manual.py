import streamlit as st
import pandas as pd
from src.utils.mlflow_utils import set_tracking
from src.serving.inference import predict_calculator

st.markdown("""
<style>
.block-container {
  max-width: 1000px;
  padding-left: 2rem;
  padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

try:
    set_tracking()
except Exception as e:
    st.error(f"Couldn't set mlflow tracking uri, check config: {e}")
    st.stop()
st.set_page_config(page_title="Property AI Lab 🏠", layout="centered")

st.title("🏠 Property Price Predictor")
st.markdown("Podaj cechy nieruchomości na sprzedaz w Katowicach, a model oszacuje jej wartość.")

def fmt_pln(x: float) -> str:
    return f"{x:,.0f} PLN".replace(",", " ")


st.subheader("Dane podstawowe")
b1, b2 = st.columns(2)
with b1:
    market = st.selectbox("Rynek", ["pierwotny", "wtórny"])
    area = st.number_input("Powierzchnia (m²)", min_value=10.0, max_value=1000.0, value=60.0, step=1.0)
with b2:
    floor_options = ['1','2','0','3','4','10+','5','6','7','8','9','10']
    rooms_num = st.number_input("Liczba pokoi", min_value=1, max_value=20, value=3, step=1)
    floor_num = st.selectbox("Piętro", floor_options, index=2)

st.subheader("Budynek")
bu1, bu2 = st.columns(2)
with bu1:
    building_floors_num = st.number_input("Liczba pięter w budynku", min_value=1, max_value=100, value=5, step=1)
    building_material = st.selectbox("Materiał budynku", ["cegła", "płyta", "beton", "inne"])
with bu2:
    building_age = st.number_input("Wiek budynku (lata)", min_value=0, max_value=200, value=10, step=1)
    construction_status = st.selectbox("Status budowy", ["gotowy", "w budowie"])

st.subheader("Lokalizacja")
district = st.selectbox(
    "Dzielnica",
    ['Śródmieście', 'Wełnowiec-Józefowiec', 'Osiedle Paderewskiego-Muchowiec', 'Bogucice',
     'Brynów-Osiedle Zgrzebnioka', 'Koszutka', 'Ligota-Panewniki', 'Osiedle Tysiąclecia',
     'Piotrowice-Ochojec', 'Dąb', 'Załęska Hałda-Brynów', 'Szopienice-Burowiec', 'Janów-Nikiszowiec',
     'Giszowiec', 'Kostuchna', 'Zawodzie', 'Osiedle Witosa', 'Załęże', 'Dąbrówka Mała', 'Murcki',
     'WPKiW', 'Podlesie', 'Zarzecze']
)

st.subheader("Udogodnienia")
u1, u2, u3, u4 = st.columns(4)
with u1:
    balcony = st.selectbox("Balkon", ["nie", "tak"])
    terrace = st.selectbox("Taras", ["nie", "tak"])
    garden = st.selectbox("Ogród", ["nie", "tak"])
    garage = st.selectbox("Garaż", ["nie", "tak"])
with u2:
    internet = st.selectbox("Internet", ["nie", "tak"])
    lift = st.selectbox("Winda", ["nie", "tak"])
    separate_kitchen = st.selectbox("Oddzielna kuchnia", ["nie", "tak"])
    dishwasher = st.selectbox("Zmywarka", ["nie", "tak"])
with u3:
    air_conditioning = st.selectbox("Klimatyzacja", ["nie", "tak"])
    roller_shutters = st.selectbox("Rolety", ["nie", "tak"])
    anti_burglary_door = st.selectbox("Drzwi antywłamaniowe", ["nie", "tak"])
    closed_area = st.selectbox("Osiedle zamknięte", ["nie", "tak"])
with u4:
    entryphone = st.selectbox("Domofon / wideofon", ["nie", "tak"])
    monitoring = st.selectbox("Monitoring", ["nie", "tak"])
    alarm = st.selectbox("Alarm", ["nie", "tak"])
    basement = st.selectbox("Piwnica", ["nie", "tak"])

input_df = pd.DataFrame([{
    "market": market,
    "building_age": building_age,
    "area": area,
    "rooms_num": rooms_num,
    "building_floors_num": building_floors_num,
    "floor_num": floor_num,
    "building_material": building_material,
    "construction_status": construction_status,
    "district": district,
    "balcony": balcony,
    "separate_kitchen": separate_kitchen,
    "air_conditioning": air_conditioning,
    "roller_shutters": roller_shutters,
    "dishwasher": dishwasher,
    "garage": garage,
    "anti_burglary_door": anti_burglary_door,
    "basement": basement,
    "entryphone": entryphone,
    "garden": garden,
    "internet": internet,
    "monitoring": monitoring,
    "terrace": terrace,
    "alarm": alarm,
    "lift": lift,
    "closed_area": closed_area,
}])

st.markdown("---")
if st.button("🔮 Przewiduj cenę"):
    yhat = predict_calculator(input_df)
    total_price = float(yhat[0])
    price_per_m2 = total_price / float(area)

    colA, colB = st.columns(2)
    with colA:
        st.metric("Cena za m² (prognoza)", fmt_pln(price_per_m2))
    with colB:
        st.metric("Cena całkowita (prognoza)", fmt_pln(total_price))

    with st.expander("Podgląd danych wejściowych"):
        st.dataframe(input_df)

st.caption("Property AI Lab | Streamlit + MLflow | demo")
