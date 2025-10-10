import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb

st.markdown("""
<style>
.block-container {
  max-width: 1000px; 
  padding-left: 2rem;
  padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

mlflow.set_tracking_uri("http://127.0.0.1:5002")  

st.set_page_config(page_title="Property AI Lab 🏠", layout="centered")

st.title("🏠 Property Price Predictor")
st.markdown("Podaj cechy nieruchomości na sprzedaz w Katowicach, a model oszacuje jej wartość.")

MODEL_URI = "runs:/785968a6a7ac46b1916321c9ae853c8d/model"
booster = mlflow.xgboost.load_model(MODEL_URI)

FEATURES = [
    "market","building_age","area","rooms_num","building_floors_num","floor_num",
    "building_material","construction_status","district","balcony","separate_kitchen",
    "air_conditioning","roller_shutters","dishwasher","garage","anti_burglary_door",
    "basement","entryphone","garden","internet","monitoring","terrace","alarm","lift","closed_area"
]
CAT_STR_COLS = ["market","building_material","construction_status","district","floor_num"]

BIN_COLS = ["balcony","separate_kitchen","air_conditioning","roller_shutters","dishwasher",
            "garage","anti_burglary_door","basement","entryphone","garden","internet",
            "monitoring","terrace","alarm","lift","closed_area"]

INT_COLS = ["building_age"]
FLOAT_COLS = ["area","rooms_num","building_floors_num"]

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


input_df[BIN_COLS] = (input_df[BIN_COLS]
                      .replace({"tak": 1, "nie": 0, True: 1, False: 0})
                      .astype("int64"))

for c in INT_COLS:
    input_df[c] = pd.to_numeric(input_df[c], errors="coerce").fillna(0).astype("int64")
for c in FLOAT_COLS:
    input_df[c] = pd.to_numeric(input_df[c], errors="coerce").astype("float64")

for c in CAT_STR_COLS:
    input_df[c] = input_df[c].astype("category")

input_df = input_df[FEATURES]

st.markdown("---")
if st.button("🔮 Przewiduj cenę"):
    dtest = xgb.DMatrix(input_df, enable_categorical=True)
    yhat = booster.predict(dtest)
    price_per_m2 = float(np.ravel(yhat)[0])

    total_price = price_per_m2 * float(area)

    colA, colB = st.columns(2)
    with colA:
        st.metric("Cena za m² (prognoza)", fmt_pln(price_per_m2))
    with colB:
        st.metric("Cena całkowita (prognoza)", fmt_pln(total_price))

    with st.expander("Podgląd danych wejściowych"):
        st.dataframe(input_df)


st.caption("Property AI Lab | Streamlit + MLflow | demo")
