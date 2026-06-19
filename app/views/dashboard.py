import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from app.utils import is_deal
from src.data.database.db_setup import Database
from src.data.database.service import DataService
from src.serving.inference import predict


DISTRICT_CENTROIDS = {
    "śródmieście": (50.2599, 19.0211),
    "bogucice": (50.2657, 19.0463),
    "koszutka": (50.2727, 19.0247),
    "dąb": (50.2750, 19.0020),
    "wełnowiec-józefowiec": (50.2814, 19.0119),
    "załęże": (50.2670, 18.9886),
    "osiedle witosa": (50.2725, 18.9656),
    "osiedle tysiąclecia": (50.2778, 18.9856),
    "załęska hałda-brynów": (50.2425, 18.9886),
    "brynów-osiedle zgrzebnioka": (50.2389, 19.0086),
    "ligota-panewniki": (50.2289, 18.9759),
    "piotrowice-ochojec": (50.2178, 18.9967),
    "zarzecze": (50.1789, 19.0290),
    "kostuchna": (50.1860, 19.0102),
    "podlesie": (50.1620, 19.0240),
    "murcki": (50.1772, 19.0531),
    "giszowiec": (50.2308, 19.0793),
    "janów-nikiszowiec": (50.2522, 19.0772),
    "szopienice-burowiec": (50.2589, 19.0922),
    "dąbrówka mała": (50.2700, 19.0788),
    "paderewskiego-muchowiec": (50.2519, 19.0392),
}


@st.cache_data(show_spinner=True)
def load_offers(period: str) -> pd.DataFrame:
    db = Database()
    return DataService(db).load_data_by_period(period)


def aggregate_by_district(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["district_key"] = df["district"].fillna("").str.strip().str.lower()
    df = df[df["district_key"].isin(DISTRICT_CENTROIDS)]
    df["price_per_m_f"] = pd.to_numeric(df["price_per_m"], errors="coerce")
    df["area_f"] = pd.to_numeric(df["area"], errors="coerce")
    df["is_deal_int"] = df["is_deal"].astype(int)

    agg = df.groupby("district_key", as_index=False).agg(
        count=("district_key", "size"),
        mean_ppm=("price_per_m_f", "mean"),
        mean_area=("area_f", "mean"),
        deals=("is_deal_int", "sum"),
    )
    agg["lat"] = agg["district_key"].map(lambda k: DISTRICT_CENTROIDS[k][0])
    agg["lon"] = agg["district_key"].map(lambda k: DISTRICT_CENTROIDS[k][1])
    agg["district_label"] = agg["district_key"].str.title()
    agg["mean_ppm"] = agg["mean_ppm"].round(0).astype("Int64")
    agg["mean_area"] = agg["mean_area"].round(1)
    return agg


def _gradient(t: float) -> list[int]:
    t = max(0.0, min(1.0, float(t)))
    r = 240
    g = int(220 * (1 - t) + 40 * t)
    b = int(80 * (1 - t) + 30 * t)
    return [r, g, b, 220]


def _attach_visuals(df: pd.DataFrame, value_col: str, prefix: str, target_height: float = 3000.0) -> pd.DataFrame:
    vals = df[value_col].astype(float).fillna(0.0)
    vmax = float(vals.max()) if len(vals) else 0.0
    vmin = float(vals.min()) if len(vals) else 0.0

    if vmax == 0:
        df[f"{prefix}_height"] = 0.0
    else:
        df[f"{prefix}_height"] = (vals / vmax * target_height).astype(float)

    if vmax == vmin:
        df[f"{prefix}_color"] = [_gradient(0.5)] * len(df)
    else:
        df[f"{prefix}_color"] = [_gradient((v - vmin) / (vmax - vmin)) for v in vals]
    return df


ACCENT = "#F57C00"
GOOD = "#6ECD71"
MUTED = "#999"


def _market_label(df: pd.DataFrame) -> pd.Series:
    return df["market"].map({"primary": "pierwotny", "secondary": "wtórny"}).fillna("nieznany")


_MARKET_SCALE = alt.Scale(
    domain=["pierwotny", "wtórny", "nieznany"], range=[ACCENT, GOOD, MUTED]
)


def chart_price_histogram(df: pd.DataFrame, group_by_market: bool = False) -> alt.Chart:
    data = df[["price_per_m", "market"]].copy()
    data["price_per_m"] = pd.to_numeric(data["price_per_m"], errors="coerce")
    data = data.dropna(subset=["price_per_m"])

    if not group_by_market:
        return (
            alt.Chart(data)
            .mark_bar(color=ACCENT)
            .encode(
                alt.X("price_per_m:Q", bin=alt.Bin(maxbins=30), title="Cena za m² [zł]"),
                alt.Y("count():Q", title="Liczba ofert"),
                tooltip=[alt.Tooltip("count():Q", title="Oferty")],
            )
            .properties(height=300)
        )

    data["market_label"] = _market_label(data)
    return (
        alt.Chart(data)
        .mark_bar(opacity=0.85)
        .encode(
            alt.X("price_per_m:Q", bin=alt.Bin(maxbins=30), title="Cena za m² [zł]"),
            alt.Y("count():Q", title="Liczba ofert", stack="zero"),
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=["market_label:N", alt.Tooltip("count():Q", title="Oferty")],
        )
        .properties(height=300)
    )


def chart_pred_vs_real(df: pd.DataFrame) -> alt.LayerChart:
    data = df[["price", "predicted_price", "is_deal"]].copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["predicted_price"] = pd.to_numeric(data["predicted_price"], errors="coerce")
    data = data.dropna()
    data["status"] = data["is_deal"].map({True: "okazja", False: "standard"})

    max_v = float(max(data["price"].max(), data["predicted_price"].max()))
    line_df = pd.DataFrame({"x": [0, max_v], "y": [0, max_v]})

    points = (
        alt.Chart(data)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X("price:Q", title="Cena rzeczywista [zł]"),
            y=alt.Y("predicted_price:Q", title="Cena przewidziana [zł]"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["potencjalna okazja", "standard"], range=[GOOD, MUTED]),
                legend=alt.Legend(title="Status"),
            ),
            tooltip=[
                alt.Tooltip("price:Q", title="Rzeczywista", format=",.0f"),
                alt.Tooltip("predicted_price:Q", title="Predykcja", format=",.0f"),
                alt.Tooltip("status:N", title="Status"),
            ],
        )
    )
    diagonal = (
        alt.Chart(line_df)
        .mark_line(color="#bbb", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )
    return (diagonal + points).properties(height=300)


def chart_district_rank(agg: pd.DataFrame, value_col: str, axis_title: str, top_n: int = 10) -> alt.Chart:
    top = agg.nlargest(top_n, value_col)[["district_label", value_col]].copy()
    return (
        alt.Chart(top)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X(f"{value_col}:Q", title=axis_title),
            y=alt.Y("district_label:N", sort="-x", title=""),
            tooltip=["district_label:N", alt.Tooltip(f"{value_col}:Q", format=",.0f")],
        )
        .properties(height=320)
    )


def chart_build_year(df: pd.DataFrame, group_by_market: bool = False) -> alt.Chart:
    data = df[["building_build_year", "market"]].copy()
    data["building_build_year"] = pd.to_numeric(data["building_build_year"], errors="coerce")
    data = data.dropna(subset=["building_build_year"])
    data = data[(data["building_build_year"] >= 1900) & (data["building_build_year"] <= 2030)]
    data["decade"] = (data["building_build_year"] // 10 * 10).astype(int).astype(str) + "s"

    if not group_by_market:
        return (
            alt.Chart(data)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("decade:N", title="Dekada budowy"),
                y=alt.Y("count():Q", title="Liczba ofert"),
                tooltip=[alt.Tooltip("count():Q", title="Oferty")],
            )
            .properties(height=300)
        )

    data["market_label"] = _market_label(data)
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("decade:N", title="Dekada budowy"),
            y=alt.Y("count():Q", title="Liczba ofert", stack="zero"),
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=["market_label:N", alt.Tooltip("count():Q", title="Oferty")],
        )
        .properties(height=300)
    )


def chart_rooms(df: pd.DataFrame, group_by_market: bool = False) -> alt.Chart:
    data = df[["rooms_num", "market"]].copy()
    data["rooms_num"] = pd.to_numeric(data["rooms_num"], errors="coerce")
    data = data.dropna(subset=["rooms_num"])
    data["rooms_label"] = data["rooms_num"].astype(int).clip(upper=6).astype(str)
    data.loc[data["rooms_num"] >= 6, "rooms_label"] = "6+"

    if not group_by_market:
        counts = data.groupby("rooms_label", as_index=False).size().rename(columns={"size": "count"})
        return (
            alt.Chart(counts)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("rooms_label:N", title="Liczba pokoi"),
                y=alt.Y("count:Q", title="Liczba ofert"),
                tooltip=["rooms_label:N", "count:Q"],
            )
            .properties(height=300)
        )

    data["market_label"] = _market_label(data)
    counts = data.groupby(["rooms_label", "market_label"], as_index=False).size().rename(columns={"size": "count"})
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("rooms_label:N", title="Liczba pokoi"),
            y=alt.Y("count:Q", title="Liczba ofert", stack="zero"),
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=["rooms_label:N", "market_label:N", "count:Q"],
        )
        .properties(height=300)
    )


def chart_market_split(df: pd.DataFrame) -> alt.Chart:
    mapped = (
        df["market"]
        .map({"primary": "pierwotny", "secondary": "wtórny"})
        .fillna("nieznany")
    )
    counts = mapped.value_counts().reset_index()
    counts.columns = ["market", "count"]
    return (
        alt.Chart(counts)
        .mark_arc(innerRadius=60)
        .encode(
            theta="count:Q",
            color=alt.Color(
                "market:N",
                scale=alt.Scale(
                    domain=["pierwotny", "wtórny", "nieznany"],
                    range=[ACCENT, GOOD, MUTED],
                ),
                legend=alt.Legend(title="Rynek"),
            ),
            tooltip=["market:N", "count:Q"],
        )
        .properties(height=300)
    )


DOW_PL = {0: "pn", 1: "wt", 2: "śr", 3: "czw", 4: "pt", 5: "sob", 6: "ndz"}
DOW_ORDER = ["pn", "wt", "śr", "czw", "pt", "sob", "ndz"]


def _extract_hour(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.slice(0, 2), errors="coerce")


def chart_listings_over_time(df: pd.DataFrame) -> alt.Chart:
    data = df[["creation_date", "market"]].copy()
    data["creation_date"] = pd.to_datetime(data["creation_date"], errors="coerce")
    data = data.dropna(subset=["creation_date"])
    if data.empty:
        return alt.Chart(pd.DataFrame({"date": [], "count": []})).mark_area()
    data["market_label"] = _market_label(data)
    counts = (
        data.groupby([data["creation_date"].dt.date, "market_label"])
        .size()
        .reset_index(name="count")
    )
    counts.columns = ["date", "market_label", "count"]
    return (
        alt.Chart(counts)
        .mark_area(opacity=0.75, line={"strokeWidth": 1.5})
        .encode(
            x=alt.X("date:T", title="Data dodania"),
            y=alt.Y("count:Q", title="Nowe oferty", stack="zero"),
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=[
                alt.Tooltip("date:T", title="Data"),
                alt.Tooltip("market_label:N", title="Rynek"),
                alt.Tooltip("count:Q", title="Oferty"),
            ],
        )
        .properties(height=280)
    )


def chart_price_over_time(df: pd.DataFrame) -> alt.Chart:
    data = df[["creation_date", "price_per_m", "market"]].copy()
    data["creation_date"] = pd.to_datetime(data["creation_date"], errors="coerce")
    data["price_per_m"] = pd.to_numeric(data["price_per_m"], errors="coerce")
    data = data.dropna(subset=["creation_date", "price_per_m"])
    if data.empty:
        return alt.Chart(pd.DataFrame({"date": [], "mean_ppm": []})).mark_line()
    data["market_label"] = _market_label(data)
    by_day = (
        data.groupby([data["creation_date"].dt.date, "market_label"])["price_per_m"]
        .mean()
        .reset_index()
    )
    by_day.columns = ["date", "market_label", "mean_ppm"]
    return (
        alt.Chart(by_day)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X("date:T", title="Data dodania"),
            y=alt.Y(
                "mean_ppm:Q",
                title="Średnia cena za m² [zł]",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=[
                alt.Tooltip("date:T", title="Data"),
                alt.Tooltip("market_label:N", title="Rynek"),
                alt.Tooltip("mean_ppm:Q", title="Średnia cena/m²", format=",.0f"),
            ],
        )
        .properties(height=280)
    )


def chart_offers_by_dow(df: pd.DataFrame) -> alt.Chart:
    data = df[["creation_date", "market"]].copy()
    data["creation_date"] = pd.to_datetime(data["creation_date"], errors="coerce")
    data = data.dropna(subset=["creation_date"])
    if data.empty:
        return alt.Chart(pd.DataFrame({"dow": [], "count": []})).mark_bar()
    data["dow"] = data["creation_date"].dt.dayofweek.map(DOW_PL)
    data["market_label"] = _market_label(data)
    counts = data.groupby(["dow", "market_label"], as_index=False).size().rename(columns={"size": "count"})
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("dow:N", sort=DOW_ORDER, title="Dzień tygodnia"),
            y=alt.Y("count:Q", title="Liczba ofert", stack="zero"),
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=["dow:N", "market_label:N", "count:Q"],
        )
        .properties(height=280)
    )


def chart_offers_by_hour(df: pd.DataFrame) -> alt.Chart:
    data = df[["creation_time", "market"]].copy()
    data["hour"] = _extract_hour(data["creation_time"])
    data = data.dropna(subset=["hour"])
    if data.empty:
        return alt.Chart(pd.DataFrame({"hour": [], "count": []})).mark_bar()
    data["hour"] = data["hour"].astype(int)
    data["market_label"] = _market_label(data)
    counts = data.groupby(["hour", "market_label"], as_index=False).size().rename(columns={"size": "count"})
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("hour:O", title="Godzina dodania (0–23)"),
            y=alt.Y("count:Q", title="Liczba ofert", stack="zero"),
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=["hour:O", "market_label:N", "count:Q"],
        )
        .properties(height=280)
    )


def chart_dow_hour_heatmap(df: pd.DataFrame) -> alt.Chart:
    data = df[["creation_date", "creation_time"]].copy()
    data["creation_date"] = pd.to_datetime(data["creation_date"], errors="coerce")
    data["hour"] = _extract_hour(data["creation_time"])
    data = data.dropna(subset=["creation_date", "hour"])
    if data.empty:
        return alt.Chart(pd.DataFrame({"dow": [], "hour": [], "count": []})).mark_rect()
    data["hour"] = data["hour"].astype(int)
    data["dow"] = data["creation_date"].dt.dayofweek.map(DOW_PL)
    counts = data.groupby(["dow", "hour"], as_index=False).size().rename(columns={"size": "count"})
    return (
        alt.Chart(counts)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Godzina"),
            y=alt.Y("dow:N", sort=DOW_ORDER, title="Dzień"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Oferty")),
            tooltip=["dow:N", "hour:O", "count:Q"],
        )
        .properties(height=220)
    )


def chart_price_boxplot_per_district(df: pd.DataFrame) -> alt.Chart:
    data = df[["district", "price_per_m"]].copy()
    data["district"] = data["district"].fillna("").str.strip().str.title()
    data["price_per_m"] = pd.to_numeric(data["price_per_m"], errors="coerce")
    data = data.dropna(subset=["price_per_m"])
    data = data[data["district"] != ""]
    if data.empty:
        return alt.Chart(pd.DataFrame({"district": [], "price_per_m": []})).mark_boxplot()
    medians = data.groupby("district")["price_per_m"].median().sort_values(ascending=False)
    return (
        alt.Chart(data)
        .mark_boxplot(extent="min-max", color=ACCENT, size=14)
        .encode(
            x=alt.X("price_per_m:Q", title="Cena za m² [zł]"),
            y=alt.Y("district:N", sort=medians.index.tolist(), title=""),
        )
        .properties(height=max(300, 28 * len(medians)))
    )


def chart_price_vs_area(df: pd.DataFrame) -> alt.Chart:
    data = df[["price", "area", "rooms_num"]].copy()
    for c in ["price", "area", "rooms_num"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["price", "area"])
    data["rooms_label"] = "?"
    rooms_int = data["rooms_num"].fillna(0).astype(int).clip(upper=6)
    data.loc[data["rooms_num"].notna(), "rooms_label"] = rooms_int[data["rooms_num"].notna()].astype(str)
    data.loc[data["rooms_num"] >= 6, "rooms_label"] = "6+"
    return (
        alt.Chart(data)
        .mark_circle(size=50, opacity=0.6)
        .encode(
            x=alt.X("area:Q", title="Metraż [m²]"),
            y=alt.Y("price:Q", title="Cena [zł]"),
            color=alt.Color(
                "rooms_label:N",
                title="Pokoje",
                scale=alt.Scale(scheme="oranges"),
                sort=["1", "2", "3", "4", "5", "6+", "?"],
            ),
            tooltip=[
                alt.Tooltip("area:Q", title="Metraż", format=".0f"),
                alt.Tooltip("price:Q", title="Cena", format=",.0f"),
                alt.Tooltip("rooms_label:N", title="Pokoje"),
            ],
        )
        .properties(height=380)
    )


AREA_BINS = [0, 40, 60, 80, 100, 1e9]
AREA_LABELS = ["<40", "40–60", "60–80", "80–100", "100+"]


def chart_price_by_decade(df: pd.DataFrame, group_by_market: bool = False) -> alt.Chart:
    data = df[["building_build_year", "price_per_m", "market"]].copy()
    data["building_build_year"] = pd.to_numeric(data["building_build_year"], errors="coerce")
    data["price_per_m"] = pd.to_numeric(data["price_per_m"], errors="coerce")
    data = data.dropna(subset=["building_build_year", "price_per_m"])
    data = data[(data["building_build_year"] >= 1900) & (data["building_build_year"] <= 2030)]
    data["decade"] = (data["building_build_year"] // 10 * 10).astype(int).astype(str) + "s"

    if not group_by_market:
        agg = data.groupby("decade", as_index=False).agg(
            mean_ppm=("price_per_m", "mean"),
            count=("price_per_m", "size"),
        )
        return (
            alt.Chart(agg)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("decade:N", title="Dekada budowy"),
                y=alt.Y("mean_ppm:Q", title="Średnia cena za m² [zł]"),
                tooltip=[
                    alt.Tooltip("decade:N", title="Dekada"),
                    alt.Tooltip("mean_ppm:Q", title="Średnia cena/m²", format=",.0f"),
                    alt.Tooltip("count:Q", title="Oferty"),
                ],
            )
            .properties(height=300)
        )

    data["market_label"] = _market_label(data)
    agg = data.groupby(["decade", "market_label"], as_index=False).agg(
        mean_ppm=("price_per_m", "mean"),
        count=("price_per_m", "size"),
    )
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("decade:N", title="Dekada budowy"),
            y=alt.Y("mean_ppm:Q", title="Średnia cena za m² [zł]"),
            xOffset="market_label:N",
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=[
                alt.Tooltip("decade:N", title="Dekada"),
                alt.Tooltip("market_label:N", title="Rynek"),
                alt.Tooltip("mean_ppm:Q", title="Średnia cena/m²", format=",.0f"),
                alt.Tooltip("count:Q", title="Oferty"),
            ],
        )
        .properties(height=300)
    )


def chart_price_by_rooms(df: pd.DataFrame, group_by_market: bool = False) -> alt.Chart:
    data = df[["rooms_num", "price_per_m", "market"]].copy()
    data["rooms_num"] = pd.to_numeric(data["rooms_num"], errors="coerce")
    data["price_per_m"] = pd.to_numeric(data["price_per_m"], errors="coerce")
    data = data.dropna(subset=["rooms_num", "price_per_m"])
    data["rooms_label"] = data["rooms_num"].astype(int).clip(upper=6).astype(str)
    data.loc[data["rooms_num"] >= 6, "rooms_label"] = "6+"

    if not group_by_market:
        agg = data.groupby("rooms_label", as_index=False).agg(
            mean_ppm=("price_per_m", "mean"),
            count=("price_per_m", "size"),
        )
        return (
            alt.Chart(agg)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("rooms_label:N", title="Liczba pokoi"),
                y=alt.Y("mean_ppm:Q", title="Średnia cena za m² [zł]"),
                tooltip=[
                    alt.Tooltip("rooms_label:N", title="Pokoje"),
                    alt.Tooltip("mean_ppm:Q", title="Średnia cena/m²", format=",.0f"),
                    alt.Tooltip("count:Q", title="Oferty"),
                ],
            )
            .properties(height=300)
        )

    data["market_label"] = _market_label(data)
    agg = data.groupby(["rooms_label", "market_label"], as_index=False).agg(
        mean_ppm=("price_per_m", "mean"),
        count=("price_per_m", "size"),
    )
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("rooms_label:N", title="Liczba pokoi"),
            y=alt.Y("mean_ppm:Q", title="Średnia cena za m² [zł]"),
            xOffset="market_label:N",
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=[
                alt.Tooltip("rooms_label:N", title="Pokoje"),
                alt.Tooltip("market_label:N", title="Rynek"),
                alt.Tooltip("mean_ppm:Q", title="Średnia cena/m²", format=",.0f"),
                alt.Tooltip("count:Q", title="Oferty"),
            ],
        )
        .properties(height=300)
    )


def chart_price_by_area_bucket(df: pd.DataFrame, group_by_market: bool = False) -> alt.Chart:
    data = df[["area", "price_per_m", "market"]].copy()
    data["area"] = pd.to_numeric(data["area"], errors="coerce")
    data["price_per_m"] = pd.to_numeric(data["price_per_m"], errors="coerce")
    data = data.dropna(subset=["area", "price_per_m"])
    data["area_bucket"] = pd.cut(data["area"], bins=AREA_BINS, labels=AREA_LABELS, right=False)
    data = data.dropna(subset=["area_bucket"])
    data["area_bucket"] = data["area_bucket"].astype(str)

    if not group_by_market:
        agg = data.groupby("area_bucket", as_index=False).agg(
            mean_ppm=("price_per_m", "mean"),
            count=("price_per_m", "size"),
        )
        return (
            alt.Chart(agg)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("area_bucket:N", title="Metraż [m²]", sort=AREA_LABELS),
                y=alt.Y("mean_ppm:Q", title="Średnia cena za m² [zł]"),
                tooltip=[
                    alt.Tooltip("area_bucket:N", title="Metraż"),
                    alt.Tooltip("mean_ppm:Q", title="Średnia cena/m²", format=",.0f"),
                    alt.Tooltip("count:Q", title="Oferty"),
                ],
            )
            .properties(height=300)
        )

    data["market_label"] = _market_label(data)
    agg = data.groupby(["area_bucket", "market_label"], as_index=False).agg(
        mean_ppm=("price_per_m", "mean"),
        count=("price_per_m", "size"),
    )
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("area_bucket:N", title="Metraż [m²]", sort=AREA_LABELS),
            y=alt.Y("mean_ppm:Q", title="Średnia cena za m² [zł]"),
            xOffset="market_label:N",
            color=alt.Color("market_label:N", scale=_MARKET_SCALE, legend=alt.Legend(title="Rynek")),
            tooltip=[
                alt.Tooltip("area_bucket:N", title="Metraż"),
                alt.Tooltip("market_label:N", title="Rynek"),
                alt.Tooltip("mean_ppm:Q", title="Średnia cena/m²", format=",.0f"),
                alt.Tooltip("count:Q", title="Oferty"),
            ],
        )
        .properties(height=300)
    )


def chart_price_boxplot_by_rooms(df: pd.DataFrame) -> alt.Chart:
    data = df[["rooms_num", "price_per_m"]].copy()
    data["rooms_num"] = pd.to_numeric(data["rooms_num"], errors="coerce")
    data["price_per_m"] = pd.to_numeric(data["price_per_m"], errors="coerce")
    data = data.dropna()
    data["rooms_label"] = data["rooms_num"].astype(int).clip(upper=6).astype(str)
    data.loc[data["rooms_num"] >= 6, "rooms_label"] = "6+"
    return (
        alt.Chart(data)
        .mark_boxplot(extent="min-max", color=ACCENT, size=24)
        .encode(
            x=alt.X("rooms_label:N", title="Liczba pokoi", sort=["1", "2", "3", "4", "5", "6+"]),
            y=alt.Y("price_per_m:Q", title="Cena za m² [zł]"),
        )
        .properties(height=300)
    )


CONSTRUCTION_STATUS_PL = {
    "ready_to_use": "do zamieszkania",
    "to_completion": "do wykończenia",
    "to_renovation": "do remontu",
    "to_finishing": "do wykończenia",
}


def stats_table(df: pd.DataFrame, cat_col: str, value_col: str = "price_per_m") -> pd.DataFrame:
    data = df[[cat_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        return pd.DataFrame()
    g = data.groupby(cat_col)[value_col]
    stats = pd.DataFrame({
        "count": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "q1": g.quantile(0.25),
        "q3": g.quantile(0.75),
        "std": g.std().fillna(0),
        "min": g.min(),
        "max": g.max(),
    }).reset_index()
    stats["iqr"] = stats["q3"] - stats["q1"]
    for c in ["mean", "median", "q1", "q3", "std", "min", "max", "iqr"]:
        stats[c] = stats[c].round(0).astype("Int64")
    return stats


def boxplot_with_mean(
    df: pd.DataFrame,
    cat_col: str,
    cat_title: str,
    value_col: str = "price_per_m",
    sort_order=None,
    horizontal: bool = False,
    height: int = 320,
) -> alt.Chart:
    data = df[[cat_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        return alt.Chart(pd.DataFrame({cat_col: [], value_col: []})).mark_boxplot()

    if sort_order is None:
        medians = data.groupby(cat_col)[value_col].median().sort_values(ascending=False)
        sort_order = medians.index.tolist()

    means = data.groupby(cat_col, as_index=False)[value_col].mean()

    if horizontal:
        box = (
            alt.Chart(data)
            .mark_boxplot(extent="min-max", color=ACCENT, size=14)
            .encode(
                x=alt.X(f"{value_col}:Q", title="Cena za m² [zł]"),
                y=alt.Y(f"{cat_col}:N", sort=sort_order, title=cat_title),
            )
        )
        mean_dots = (
            alt.Chart(means)
            .mark_point(color="#222", filled=True, size=90, shape="diamond")
            .encode(
                x=alt.X(f"{value_col}:Q"),
                y=alt.Y(f"{cat_col}:N", sort=sort_order),
                tooltip=[
                    alt.Tooltip(f"{cat_col}:N", title=cat_title),
                    alt.Tooltip(f"{value_col}:Q", title="Średnia", format=",.0f"),
                ],
            )
        )
    else:
        box = (
            alt.Chart(data)
            .mark_boxplot(extent="min-max", color=ACCENT, size=22)
            .encode(
                x=alt.X(f"{cat_col}:N", sort=sort_order, title=cat_title),
                y=alt.Y(f"{value_col}:Q", title="Cena za m² [zł]"),
            )
        )
        mean_dots = (
            alt.Chart(means)
            .mark_point(color="#222", filled=True, size=90, shape="diamond")
            .encode(
                x=alt.X(f"{cat_col}:N", sort=sort_order),
                y=alt.Y(f"{value_col}:Q"),
                tooltip=[
                    alt.Tooltip(f"{cat_col}:N", title=cat_title),
                    alt.Tooltip(f"{value_col}:Q", title="Średnia", format=",.0f"),
                ],
            )
        )

    return (box + mean_dots).properties(height=height)


def display_stats_table(stats: pd.DataFrame, cat_col: str, cat_label: str, sort_by: str = "median"):
    if stats.empty:
        st.info("Brak danych do pokazania.")
        return
    view = stats.sort_values(sort_by, ascending=False).rename(columns={
        cat_col: cat_label,
        "count": "Oferty",
        "mean": "Średnia",
        "median": "Mediana",
        "q1": "Q1",
        "q3": "Q3",
        "iqr": "IQR",
        "std": "Odchylenie",
        "min": "Min",
        "max": "Max",
    })
    money_cols = ["Średnia", "Mediana", "Q1", "Q3", "IQR", "Odchylenie", "Min", "Max"]
    cfg = {col: st.column_config.NumberColumn(format="%d zł") for col in money_cols if col in view.columns}
    st.dataframe(view, use_container_width=True, hide_index=True, column_config=cfg)


def chart_top_agencies(df: pd.DataFrame, top_n: int = 10) -> alt.Chart:
    data = df["agency_name"].fillna("").str.strip()
    data = data[data != ""]
    if data.empty:
        return alt.Chart(pd.DataFrame({"agency": [], "count": []})).mark_bar()
    counts = data.value_counts().head(top_n).reset_index()
    counts.columns = ["agency", "count"]
    return (
        alt.Chart(counts)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("count:Q", title="Liczba ofert"),
            y=alt.Y("agency:N", sort="-x", title=""),
            tooltip=["agency:N", "count:Q"],
        )
        .properties(height=320)
    )


def chart_construction_status(df: pd.DataFrame) -> alt.Chart:
    data = df["construction_status"].fillna("brak danych").value_counts().reset_index()
    data.columns = ["status", "count"]
    return (
        alt.Chart(data)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("count:Q", title="Liczba ofert"),
            y=alt.Y("status:N", sort="-x", title=""),
            tooltip=["status:N", "count:Q"],
        )
        .properties(height=320)
    )


def render_top_deals(df: pd.DataFrame, top_n: int = 10):
    deals = df[df["is_deal"]].copy()
    if deals.empty:
        st.info("Brak potencjalnych i w wybranym zakresie.")
        return
    deals = deals.nlargest(top_n, "diff")
    cols_show = ["district", "area", "rooms_num", "price", "predicted_price", "diff", "offer_link"]
    cols_show = [c for c in cols_show if c in deals.columns]
    view = deals[cols_show].rename(columns={
        "district": "Dzielnica",
        "area": "Metraż",
        "rooms_num": "Pokoje",
        "price": "Cena",
        "predicted_price": "Predykcja",
        "diff": "Różnica %",
        "offer_link": "Link",
    })
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cena": st.column_config.NumberColumn(format="%d zł"),
            "Predykcja": st.column_config.NumberColumn(format="%d zł"),
            "Różnica %": st.column_config.NumberColumn(format="%.2f%%"),
            "Metraż": st.column_config.NumberColumn(format="%.0f m²"),
            "Link": st.column_config.LinkColumn(display_text="oferta"),
        },
    )


def _render(data: pd.DataFrame, height_col: str, color_col: str, tooltip_text: str):
    layer = pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position=["lon", "lat"],
        get_elevation=height_col,
        get_fill_color=color_col,
        radius=350,
        elevation_scale=1,
        pickable=True,
        extruded=True,
        auto_highlight=True,
    )
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=50.2350, longitude=19.0250, zoom=10.5, pitch=50, bearing=0
        ),
        layers=[layer],
        tooltip={"text": tooltip_text},
    )
    st.pydeck_chart(deck, use_container_width=True)


# --- UI ---

st.title("📊 Dashboard")
st.markdown("Przestrzenny przegląd ofert mieszkań **w Katowicach**")

col_period, col_market, col_compare, col_active = st.columns([2, 2, 1.4, 1.4])
with col_period:
    period_label = st.segmented_control(
        "Okres",
        options=["ostatni tydzień", "ostatni miesiąc"],
        selection_mode="single",
        default="ostatni tydzień",
    )
with col_market:
    market_label = st.segmented_control(
        "Rynek",
        options=["wszystkie", "pierwotny", "wtórny"],
        selection_mode="single",
        default="wszystkie",
    )
period_map = {"ostatni tydzień": "last_week", "ostatni miesiąc": "last_month"}
period = period_map[period_label]
market_map = {"pierwotny": "primary", "wtórny": "secondary"}
market_filter = market_map.get(market_label)

with col_compare:
    st.write("")   
    compare_markets = st.toggle(
        "Porównaj rynki",
        value=False,
        disabled=market_filter is not None,
        help="Włącz, żeby histogramy i wykresy struktury rozdzielały słupki na pierwotny/wtórny. Działa tylko przy filtrze rynku = wszystkie.",
    )
    compare_markets = compare_markets and market_filter is None

with col_active:
    st.write("")
    only_active = st.toggle(
        "Tylko aktywne",
        value=True,
        help="Odsiej oferty wycofane / zamknięte (active = False).",
    )

try:
    df_all = load_offers(period)
except Exception as e:
    st.error(f"Błąd ładowania danych z bazy: {e}")
    st.stop()

if df_all.empty:
    st.warning("Brak danych w bazie.")
    st.stop()

if "city" in df_all.columns:
    city_norm = df_all["city"].fillna("").str.strip().str.lower()
    df_all = df_all[city_norm == "katowice"].copy()

if market_filter is not None:
    df_all = df_all[df_all["market"] == market_filter].copy()
    if df_all.empty:
        st.warning(f"Brak ofert z rynku {market_label} w wybranym okresie.")
        st.stop()

if only_active and "active" in df_all.columns:
    df_all = df_all[df_all["active"].fillna(True).astype(bool)].copy()
    if df_all.empty:
        st.warning("Brak aktywnych ofert w wybranym zakresie.")
        st.stop()

df_all["predicted_price"] = predict(df_all)
diff, deals = is_deal(df_all)
df_all["diff"] = (diff * 100).round(2)
df_all["is_deal"] = deals

agg = aggregate_by_district(df_all)
matched = int(agg["count"].sum()) if not agg.empty else 0
unmatched = len(df_all) - matched

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ofert łącznie", len(df_all))
c2.metric("Na mapie", matched)
c3.metric("Potencjalnych okazji", int(df_all["is_deal"].sum()))
ppm_series = pd.to_numeric(df_all["price_per_m"], errors="coerce")
mean_ppm_all = ppm_series.mean()
median_ppm_all = ppm_series.median()
c4.metric(
    "Średnia cena/m²",
    f"{mean_ppm_all:,.0f} zł".replace(",", " ") if pd.notna(mean_ppm_all) else "—",
)
c5.metric(
    "Mediana cena/m²",
    f"{median_ppm_all:,.0f} zł".replace(",", " ") if pd.notna(median_ppm_all) else "—",
    help="Odporna na outliery — flipperzy patrzą na medianę, średnia kłamie przy ekstremalnych ofertach.",
)

if unmatched > 0:
    st.caption(f"{unmatched} ofert pominięto (dzielnica spoza słownika Katowic).")

if agg.empty:
    st.warning("Żadna oferta nie ma rozpoznawalnej dzielnicy Katowic.")
    st.stop()

agg = _attach_visuals(agg, "count", "count")
agg = _attach_visuals(agg, "mean_ppm", "price")
agg = _attach_visuals(agg, "deals", "deals")

st.markdown("### 🗺️ Mapy")
col_count, col_price, col_deals = st.columns(3)

with col_count:
    st.markdown("##### Liczba ofert")
    st.caption("Wysokość słupka proporcjonalna do liczby ofert w dzielnicy.")
    _render(agg, "count_height", "count_color", "{district_label}\nOferty: {count}")

with col_price:
    st.markdown("##### Średnia cena/m²")
    st.caption("Wysokość słupka proporcjonalna do średniej ceny za m².")
    _render(agg, "price_height", "price_color", "{district_label}\nŚrednia: {mean_ppm} zł/m²")

with col_deals:
    st.markdown("##### Liczba potencjlanych okazji")
    st.caption("Wysokość słupka proporcjonalna do liczby ofert oznaczonych jako potencjalna okazja.")
    _render(agg, "deals_height", "deals_color", "{district_label}\nPotencjalne okazje: {deals}")


st.divider()
st.markdown("### 📈 Dynamika rynku w czasie")
st.caption(
    "Czy w wybranym okresie podaż rośnie, a ceny się zmieniają. Dla flippera/analityka "
    "to puls rynku — szukaj punktów odchyleń od trendu."
)
col_dyn_a, col_dyn_b = st.columns(2)
with col_dyn_a:
    st.markdown("##### Nowe oferty per dzień")
    st.caption("Rozdzielone na pierwotny / wtórny — deweloperzy wrzucają paczkami i zniekształcają trend.")
    st.altair_chart(chart_listings_over_time(df_all), use_container_width=True)
with col_dyn_b:
    st.markdown("##### Średnia cena za m² per dzień")
    st.caption("Osobne linie dla rynków — pierwotny i wtórny rządzą się innymi cyklami cenowymi.")
    st.altair_chart(chart_price_over_time(df_all), use_container_width=True)


st.divider()
st.markdown("### 🕐 Kiedy są wrzucane oferty")
st.caption(
    "Najczęstsze dni i godziny dodania. Pośrednik widzi, kiedy reagować szybko; "
    "analityk widzi rytm aktywności agencji vs deweloperów (deweloperzy często wrzucają w godzinach pracy w pn-pt)."
)
col_when_a, col_when_b = st.columns(2)
with col_when_a:
    st.markdown("##### Dzień tygodnia")
    st.altair_chart(chart_offers_by_dow(df_all), use_container_width=True)
with col_when_b:
    st.markdown("##### Godzina dnia")
    st.altair_chart(chart_offers_by_hour(df_all), use_container_width=True)

st.markdown("##### Heatmapa dzień × godzina")
st.caption("Im ciemniej, tym więcej ofert dodawanych w tym slotcie. Pokazuje pełne okno aktywności rynku.")
st.altair_chart(chart_dow_hour_heatmap(df_all), use_container_width=True)


st.divider()
st.markdown("### 💰 Rozkład cen i predykcja modelu")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("##### Histogram ceny za m²")
    st.caption("Jak rozkładają się ceny ofertowe w wybranym zakresie.")
    st.altair_chart(
        chart_price_histogram(df_all, group_by_market=compare_markets),
        use_container_width=True,
    )
with col_b:
    st.markdown("##### Predykcja modelu vs cena rzeczywista")
    st.caption("Punkty nad linią — oferty wycenione poniżej predykcji modelu (potencjalne okazje).")
    st.altair_chart(chart_pred_vs_real(df_all), use_container_width=True)


st.divider()
st.markdown("### 📦 Rozrzut cen/m² per dzielnica")
st.caption(
    "Boxplot — mediana, kwartyle Q1/Q3, max-min. Oferty poniżej Q1 dzielnicy to potencjalne okazje, "
    "powyżej Q3 to premium. Dzielnice posortowane po medianie malejąco."
)
st.altair_chart(chart_price_boxplot_per_district(df_all), use_container_width=True)


st.divider()
st.markdown("### 🏘️ Ranking dzielnic")
rank_metric = st.segmented_control(
    "Metryka",
    options=["cena/m²", "liczba ofert", "liczba potencjalnych okazji", "średni metraż"],
    selection_mode="single",
    default="cena/m²",
    key="rank_metric",
)
rank_map = {
    "cena/m²": ("mean_ppm", "Średnia cena za m² [zł]"),
    "liczba ofert": ("count", "Liczba ofert"),
    "liczba potencjalnych okazji": ("deals", "Liczba potencjalnych okazji"),
    "średni metraż": ("mean_area", "Średni metraż [m²]"),
}
rank_col, rank_axis = rank_map[rank_metric]
st.altair_chart(
    chart_district_rank(agg, rank_col, rank_axis, top_n=15),
    use_container_width=True,
)


st.divider()
st.markdown("### 🎯 Segmentacja: cena × metraż")
st.caption(
    "Każdy punkt to oferta. Kolor — liczba pokoi. Pozwala zobaczyć segmenty rynku: kawalerki na flipa "
    "(lewy dolny), kompakty rodzinne (środek), premium (prawy górny)."
)
st.altair_chart(chart_price_vs_area(df_all), use_container_width=True)


st.divider()
st.markdown("### 💲 Średnia cena/m² wg charakterystyki mieszkania")
st.caption(
    "Jak wiek budynku, liczba pokoi i metraż wpływają na cenę za metr. "
    "Włącz 'Porównaj rynki', żeby zobaczyć różnice pierwotny vs wtórny (deweloperzy biorą premium za nowość)."
)
col_pf_a, col_pf_b, col_pf_c = st.columns(3)
with col_pf_a:
    st.markdown("##### wg dekady budowy")
    st.altair_chart(
        chart_price_by_decade(df_all, group_by_market=compare_markets),
        use_container_width=True,
    )
with col_pf_b:
    st.markdown("##### wg liczby pokoi")
    st.altair_chart(
        chart_price_by_rooms(df_all, group_by_market=compare_markets),
        use_container_width=True,
    )
with col_pf_c:
    st.markdown("##### wg przedziału metrażu")
    st.altair_chart(
        chart_price_by_area_bucket(df_all, group_by_market=compare_markets),
        use_container_width=True,
    )

st.markdown("##### Rozrzut ceny/m² wg liczby pokoi (boxplot)")
st.caption(
    "Mediana, kwartyle, outliery dla każdego segmentu pokojowego — pokazuje, "
    "gdzie w danym typie szukać ofert poniżej Q1 (potencjalne okazje)."
)
st.altair_chart(chart_price_boxplot_by_rooms(df_all), use_container_width=True)


st.divider()
st.markdown("### 📊 Średnia / mediana / rozrzut cen/m²")
st.caption(
    "Pełna statystyka — rozkład cen za m² w czterech wymiarach. "
    "Romb = średnia, środek pudełka = mediana, brzegi = Q1/Q3, wąsy = min/max. "
    "Duży IQR (rozrzut Q3-Q1) sygnalizuje segment z większą zmiennością — tam częściej trafiają się potencjalne okazje."
)

df_stat = df_all.copy()
df_stat["price_per_m_f"] = pd.to_numeric(df_stat["price_per_m"], errors="coerce")

# wiek
df_stat["building_build_year_f"] = pd.to_numeric(df_stat["building_build_year"], errors="coerce")
df_decade = df_stat[(df_stat["building_build_year_f"] >= 1900) & (df_stat["building_build_year_f"] <= 2030)].copy()
df_decade["decade"] = (df_decade["building_build_year_f"] // 10 * 10).astype(int).astype(str) + "s"

# metraz
df_stat["area_f"] = pd.to_numeric(df_stat["area"], errors="coerce")
df_area_st = df_stat.dropna(subset=["area_f"]).copy()
df_area_st["area_bucket"] = pd.cut(df_area_st["area_f"], bins=AREA_BINS, labels=AREA_LABELS, right=False).astype(str)
df_area_st = df_area_st[df_area_st["area_bucket"] != "nan"]

# dzielnica
df_dist_st = df_stat.copy()
df_dist_st["district_label"] = df_dist_st["district"].fillna("").str.strip().str.title()
df_dist_st = df_dist_st[df_dist_st["district_label"] != ""]

# stan wykonczenia
df_status_st = df_stat.copy()
df_status_st["status_label"] = (
    df_status_st["construction_status"]
    .fillna("brak danych")
    .astype(str)
    .str.strip()
    .replace(CONSTRUCTION_STATUS_PL)
    .replace({"": "brak danych"})
)

tab_decade, tab_area, tab_dist, tab_status = st.tabs([
    "🏗️ Wiek budynku",
    "📐 Metraż",
    "📍 Dzielnica",
    "🔨 Stan wykończenia",
])

with tab_decade:
    decades_sorted = sorted(df_decade["decade"].dropna().unique().tolist())
    st.altair_chart(
        boxplot_with_mean(
            df_decade, "decade", "Dekada budowy", "price_per_m_f",
            sort_order=decades_sorted, height=340,
        ),
        use_container_width=True,
    )
    display_stats_table(
        stats_table(df_decade, "decade", "price_per_m_f"),
        "decade", "Dekada",
    )

with tab_area:
    st.altair_chart(
        boxplot_with_mean(
            df_area_st, "area_bucket", "Przedział metrażu [m²]", "price_per_m_f",
            sort_order=AREA_LABELS, height=340,
        ),
        use_container_width=True,
    )
    display_stats_table(
        stats_table(df_area_st, "area_bucket", "price_per_m_f"),
        "area_bucket", "Metraż",
    )

with tab_dist:
    n_dist = df_dist_st["district_label"].nunique()
    st.altair_chart(
        boxplot_with_mean(
            df_dist_st, "district_label", "Dzielnica", "price_per_m_f",
            horizontal=True, height=max(320, 26 * n_dist),
        ),
        use_container_width=True,
    )
    display_stats_table(
        stats_table(df_dist_st, "district_label", "price_per_m_f"),
        "district_label", "Dzielnica",
    )

with tab_status:
    st.altair_chart(
        boxplot_with_mean(
            df_status_st, "status_label", "Stan wykończenia", "price_per_m_f",
            height=340,
        ),
        use_container_width=True,
    )
    display_stats_table(
        stats_table(df_status_st, "status_label", "price_per_m_f"),
        "status_label", "Stan wykończenia",
    )


st.divider()
st.markdown("### 🏢 Struktura ofert")
if market_filter is None:
    col_e, col_f, col_g = st.columns(3)
else:
    col_e, col_f = st.columns(2)
    col_g = None
with col_e:
    st.markdown("##### Wiek budynków")
    st.altair_chart(
        chart_build_year(df_all, group_by_market=compare_markets),
        use_container_width=True,
    )
with col_f:
    st.markdown("##### Liczba pokoi")
    st.altair_chart(
        chart_rooms(df_all, group_by_market=compare_markets),
        use_container_width=True,
    )
if col_g is not None:
    with col_g:
        st.markdown("##### Rynek pierwotny vs wtórny")
        st.altair_chart(chart_market_split(df_all), use_container_width=True)


st.divider()
st.markdown("### 🏷️ Aktorzy rynku")
st.caption("Kto kontroluje podaż i ile ofert jest gotowych do zamieszkania vs do remontu.")
col_act_a, col_act_b = st.columns(2)
with col_act_a:
    st.markdown("##### Top 10 agencji po liczbie ofert")
    st.altair_chart(chart_top_agencies(df_all), use_container_width=True)
with col_act_b:
    st.markdown("##### Stan wykończenia / budowy")
    st.altair_chart(chart_construction_status(df_all), use_container_width=True)


st.divider()
st.markdown("### 🔥 Top 10 potencjalnych okazji")
st.caption("Oferty z największą różnicą między predykcją a ceną ofertową (sortowane malejąco).")
render_top_deals(df_all)
