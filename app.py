import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────
st.set_page_config(page_title="ARMA Temperature Analysis", layout="wide")
st.title("ARMA Model — Indian City Temperature Analysis")
st.markdown("---")

# ── Load Data ────────────────────────────────
@st.cache_data
def load_data():
    files = {
        'Delhi'    : 'delhi.csv',
        'Mumbai'   : 'mumbai.csv',
        'Bengaluru': 'bengaluru.csv',
        'Chennai'  : 'chennai.csv',
        'Kolkata'  : 'kolkata.csv'
    }
    city_data = {}
    for city, fname in files.items():
        df = pd.read_csv(fname)
        df = df.drop(columns=['Rain'], errors='ignore')
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df['Temp Mean'] = (pd.to_numeric(df['Temp Max'], errors='coerce') +
                           pd.to_numeric(df['Temp Min'], errors='coerce')) / 2
        df = df.dropna(subset=['Date', 'Temp Mean'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()[~df.index.duplicated()]
        city_data[city] = df['Temp Mean']

    combined = pd.DataFrame(city_data).dropna()
    combined = combined[(combined < 45) & (combined > 5)].dropna()
    return combined

combined = load_data()

# ── Tabs ─────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Temperature Plots",
    "ADF Test",
    "ARMA + Results"
])

# ── TAB 1 — Data Overview ────────────────────
with tab1:
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cities", len(combined.columns))
    col2.metric("Records", f"{len(combined):,}")
    col3.metric("Date Range", f"{combined.index.min().year} – {combined.index.max().year}")
    st.dataframe(combined.describe().round(2), use_container_width=True)

# ── TAB 2 — Temperature Plots ────────────────
with tab2:
    st.header("Daily Mean Temperature (1951–2024)")
    for city in combined.columns:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(combined.index, combined[city], color='steelblue', alpha=0.7, linewidth=0.5)
        ax.set_title(f"{city}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Temp (°C)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── TAB 3 — ADF Test ─────────────────────────
with tab3:
    st.header("Stationarity Test — ADF Test")
    st.markdown("**H0:** Data is NOT stationary | **p < 0.05** → stationary ✅")

    if st.button("Run ADF Test"):
        rows = []
        for city in combined.columns:
            result = adfuller(combined[city].dropna())
            rows.append({
                "City"         : city,
                "ADF Statistic": round(result[0], 4),
                "p-value"      : f"{result[1]:.2e}",
                "Stationary?"  : "Yes ✅" if result[1] < 0.05 else "No ❌"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── TAB 4 — ARMA + Results ───────────────────
with tab4:
    st.header("ARMA Model — Grid Search + MAE Evaluation")
    st.markdown("Train: **1951–2020** | Test: **2021–2024**")

    if st.button("Run ARMA Grid Search"):
        results = []
        progress = st.progress(0)

        for i, city in enumerate(combined.columns):
            ts    = combined[city].dropna()
            train = ts[ts.index < '2021-01-01']
            test  = ts[ts.index >= '2021-01-01']

            # baseline
            base     = ARIMA(train, order=(1, 0, 1)).fit()
            base_mae = mean_absolute_error(test, base.predict(start=len(train), end=len(ts)-1))

            # grid search
            best_mae, best_order = float('inf'), (1, 1)
            for p, q in itertools.product(range(4), range(4)):
                try:
                    m   = ARIMA(train, order=(p, 0, q)).fit()
                    mae = mean_absolute_error(test, m.predict(start=len(train), end=len(ts)-1))
                    if mae < best_mae:
                        best_mae, best_order = mae, (p, q)
                except:
                    continue

            results.append({
                "City"          : city,
                "ARMA(1,1) MAE" : round(base_mae, 2),
                "Best Order"    : f"ARMA{best_order}",
                "Best MAE"      : round(best_mae, 2),
                "Improvement"   : round(base_mae - best_mae, 2)
            })
            progress.progress((i + 1) / len(combined.columns))

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(df))
        ax.bar([i - 0.2 for i in x], df["ARMA(1,1) MAE"], 0.4, label="Baseline", color="steelblue")
        ax.bar([i + 0.2 for i in x], df["Best MAE"],       0.4, label="Best",     color="tomato")
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["City"])
        ax.set_ylabel("MAE (°C)")
        ax.set_title("Baseline vs Best Model MAE")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()