import os
import sys
import pathlib

# entry sits in app/ so the project root isn't on sys.path by default
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# streamlit cloud puts secrets in st.secrets, not in os.environ
# copy them across so the rest of the app keeps reading config from one place (.env locally, streamlit cloud secrets in prod)
try:
    for _k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
               "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT", "DB_SSLMODE"):
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from src.utils.mlflow_utils import set_tracking


st.set_page_config(page_title="Property AI Lab 🏠", page_icon="🏠", layout="wide")

st.markdown("""
<style>
/* Top header background + bigger height */
header[data-testid="stHeader"] {
  background: linear-gradient(180deg, #f5f7fb 0%, #eef1f7 100%);
  border-bottom: 1px solid #d8dde6;
  height: 64px;
}

header[data-testid="stHeader"] [data-testid="stPageLink-NavLink"]:hover,
header[data-testid="stHeader"] a[role="link"]:hover {
  background-color: rgba(0, 0, 0, 0.04);
}

</style>
""", unsafe_allow_html=True)

try:
    set_tracking()
except Exception as e:
    st.error(f"Couldn't set mlflow tracking uri, check config: {e}")
    st.stop()


offers = st.Page("views/offers.py", title="Oferty", icon="🏠", default=True)
calculator = st.Page("views/calculator.py", title="Kalkulator", icon="🧮")

pg = st.navigation([offers, calculator], position="top")
pg.run()
