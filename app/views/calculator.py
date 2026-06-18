import streamlit as st
import pandas as pd

from src.serving.inference import predict


st.markdown("""
<style>
.block-container {
  max-width: 1000px;
  padding-left: 2rem;
  padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 Property Price Predictor")
st.markdown("Podaj cechy nieruchomości na sprzedaz w Katowicach, a model oszacuje jej wartość.")


def fmt_pln(x: float) -> str:
    return f"{x:,.0f} PLN".replace(",", " ")


MARKET_LABELS = {
    "primary": "pierwotny",
    "secondary": "wtórny",
}
ADVERT_TYPE_LABELS = {
    "developer_unit": "deweloper",
    "agency": "agencja",
    "private": "prywatna",
}
BUILDING_MATERIAL_LABELS = {
    "brick": "cegła",
    "concrete_plate": "płyta",
    "concrete": "beton",
    "reinforced_concrete": "żelbet",
    "other": "inne",
}
BUILDING_TYPE_LABELS = {
    "block": "blok",
    "apartment": "apartamentowiec",
    "tenement": "kamienica",
    "other": "inne",
}
CONSTRUCTION_STATUS_LABELS = {
    "ready_to_use": "gotowy",
    "to_completion": "w budowie",
    "to_renovation": "do remontu",
}
YES_NO_LABELS = {False: "nie", True: "tak"}


def yes_no(label):
    return st.selectbox(label, [False, True], format_func=YES_NO_LABELS.get)


st.subheader("Dane podstawowe")
b1, b2 = st.columns(2)
with b1:
    market = st.selectbox("Rynek", list(MARKET_LABELS), format_func=MARKET_LABELS.get)
    advert_type = st.selectbox("Typ ogłoszenia", list(ADVERT_TYPE_LABELS), format_func=ADVERT_TYPE_LABELS.get)
    area = st.number_input("Powierzchnia (m²)", min_value=10.0, max_value=1000.0, value=60.0, step=1.0)
with b2:
    floor_options = ['1','2','0','3','4','10+','5','6','7','8','9','10']
    rooms_num = st.number_input("Liczba pokoi", min_value=1, max_value=20, value=3, step=1)
    floor_num = st.selectbox("Piętro", floor_options, index=2)

st.subheader("Budynek")
bu1, bu2 = st.columns(2)
with bu1:
    building_floors_num = st.number_input("Liczba pięter w budynku", min_value=1, max_value=100, value=5, step=1)
    building_material = st.selectbox(
        "Materiał budynku",
        list(BUILDING_MATERIAL_LABELS),
        format_func=BUILDING_MATERIAL_LABELS.get,
    )
    building_type = st.selectbox("Typ budynku", list(BUILDING_TYPE_LABELS), format_func=BUILDING_TYPE_LABELS.get)
with bu2:
    building_age = st.number_input("Wiek budynku (lata)", min_value=0, max_value=200, value=10, step=1)
    construction_status = st.selectbox(
        "Status budowy",
        list(CONSTRUCTION_STATUS_LABELS),
        format_func=CONSTRUCTION_STATUS_LABELS.get,
    )

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
    balcony = yes_no("Balkon")
    terrace = yes_no("Taras")
    garden = yes_no("Ogród")
    garage = yes_no("Garaż")
with u2:
    lift = yes_no("Winda")
    separate_kitchen = yes_no("Oddzielna kuchnia")
    dishwasher = yes_no("Zmywarka")
    furniture = yes_no("Meble")
with u3:
    anti_burglary_door = yes_no("Drzwi antywłamaniowe")
    closed_area = yes_no("Osiedle zamknięte")
    entryphone = yes_no("Domofon / wideofon")
    monitoring = yes_no("Monitoring")
with u4:
    basement = yes_no("Piwnica")
    usable_room = yes_no("Pomieszczenie użytkowe")

input_df = pd.DataFrame([{
    "market": market,
    "advert_type": advert_type,
    "building_age": building_age,
    "area": area,
    "rooms_num": rooms_num,
    "building_floors_num": building_floors_num,
    "floor_num": floor_num,
    "building_material": building_material,
    "building_type": building_type,
    "construction_status": construction_status,
    "district": district,
    "balcony": balcony,
    "separate_kitchen": separate_kitchen,
    "dishwasher": dishwasher,
    "garage": garage,
    "anti_burglary_door": anti_burglary_door,
    "basement": basement,
    "entryphone": entryphone,
    "garden": garden,
    "monitoring": monitoring,
    "terrace": terrace,
    "lift": lift,
    "closed_area": closed_area,
    "furniture": furniture,
    "usable_room": usable_room,
}])

st.markdown("---")
if st.button("🔮 Przewiduj cenę"):
    yhat = predict(input_df)
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
