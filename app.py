import importlib.util
import logging
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Fuel Consumption Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------------
# The Fuel_Log sheet has: Date, Supplier, Department, Truck_Reg,
# Unit_Price_KSH/L, Quantity_Ltrs, Amount_KSH. The dashboard works with the
# names on the right, so rename on load.
COLUMN_MAP = {
    "Supplier": "Fuel_Supplier",
    "Quantity_Ltrs": "Litres",
    "Amount_KSH": "Cost",
    "Amount": "Cost",
}
REQUIRED = ["Date", "Department", "Truck_Reg", "Fuel_Supplier", "Litres", "Cost"]


def parse_dates(s: pd.Series) -> pd.Series:
    """Fuel_Log dates are dd.mm.yy; fall back to day-first parsing otherwise."""
    out = pd.to_datetime(s, format="%d.%m.%y", errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        out[bad] = pd.to_datetime(s[bad], dayfirst=True, errors="coerce")
    return out


def clean_fuel_log(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns=COLUMN_MAP)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s): {', '.join(missing)}. "
            f"Columns found: {', '.join(df.columns)}"
        )

    df["Date"] = parse_dates(df["Date"])
    for c in ("Litres", "Cost"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED)
    df = df[df["Litres"] > 0].copy()
    for c in ("Department", "Truck_Reg", "Fuel_Supplier"):
        df[c] = df[c].astype(str).str.strip()

    df["Cost_per_Litre"] = df["Cost"] / df["Litres"]
    return df.set_index("Date").sort_index()


@st.cache_data(ttl=300, show_spinner="Loading data...")
def load_fuel_log(source) -> pd.DataFrame:
    """source = uploaded file bytes, or a CSV URL."""
    buf = BytesIO(source) if isinstance(source, bytes) else source
    return clean_fuel_log(pd.read_csv(buf, thousands=","))


def simulated_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2023-01-01", periods=24, freq="MS")
    df = pd.DataFrame({
        "Date": dates,
        "Department": rng.choice(["Transport", "Logistics", "Sales"], size=24),
        "Truck_Reg": rng.choice(["KAA123X", "KBB456Y", "KCC789Z"], size=24),
        "Fuel_Supplier": rng.choice(["Shell", "Total", "Rubis"], size=24),
        "Litres": rng.integers(200, 800, 24),
        "Cost": rng.integers(30000, 80000, 24),
    })
    df["Cost_per_Litre"] = df["Cost"] / df["Litres"]
    return df.set_index("Date")


def add_anomaly_flags(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Z-score of Litres within each department (safe when std is 0 or n = 1)."""
    def z(x):
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd > 0 else x * 0.0

    df = df.copy()
    df["Consumption_ZScore"] = df.groupby("Department")["Litres"].transform(z)
    df["Anomaly"] = np.where(df["Consumption_ZScore"] > threshold, "🔴 High", "")
    return df


# ---------------------------------------------------------------------------
# Forecasting (only runs when the button is pressed, and is cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fitting forecasts...")
def run_forecasts(data: pd.DataFrame, trucks: tuple, freq: str, horizon: int, min_points: int):
    from prophet import Prophet  # imported here so the rest of the app works without it

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    results, too_short, failed = [], [], []
    for truck in trucks:
        t = data[data["Truck_Reg"] == truck]
        for metric in ("Litres", "Cost"):
            # Total per period; periods with no fuel bought count as 0
            series = t[metric].resample(freq).sum()
            if len(series) < min_points:
                if truck not in too_short:
                    too_short.append(truck)
                continue
            hist = series.rename("y").rename_axis("ds").reset_index()
            try:
                model = Prophet(yearly_seasonality=False, daily_seasonality=False)
                model.fit(hist)
                future = model.make_future_dataframe(periods=horizon, freq=freq)
                fc = model.predict(future)[["ds", "yhat"]].tail(horizon).copy()
                fc["yhat"] = fc["yhat"].clip(lower=0)
                fc["Truck_Reg"] = truck
                fc["Metric"] = metric
                results.append(fc)
            except Exception as e:
                failed.append(f"{truck} ({metric}): {e}")

    combined = None
    if results:
        combined = pd.concat(results, ignore_index=True).rename(
            columns={"ds": "Date", "yhat": "Forecasted_Value"}
        )
    return combined, too_short, failed


def to_excel(sheets: dict) -> bytes:
    engine = "xlsxwriter" if importlib.util.find_spec("xlsxwriter") else "openpyxl"
    out = BytesIO()
    with pd.ExcelWriter(out, engine=engine) as writer:
        for name, d in sheets.items():
            if d is not None and not d.empty:
                d.to_excel(writer, sheet_name=name, index=False)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Data input
# ---------------------------------------------------------------------------
st.sidebar.title("📥 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Fuel_Log CSV", type=["csv"])
sheet_url = st.sidebar.text_input(
    "...or Fuel_Log CSV link (optional)",
    help="For a sheet shared as 'Anyone with the link': "
         "https://docs.google.com/spreadsheets/d/<SHEET_ID>/gviz/tq?tqx=out:csv&sheet=Fuel_Log",
)

try:
    if uploaded_file is not None:
        df_all = load_fuel_log(uploaded_file.getvalue())
        st.success(f"✅ Loaded {len(df_all):,} fuel records.")
    elif sheet_url.strip():
        df_all = load_fuel_log(sheet_url.strip())
        st.success(f"✅ Loaded {len(df_all):,} fuel records from the link.")
    else:
        st.sidebar.info("No data source given. Using simulated data.")
        df_all = simulated_data()
except Exception as e:
    st.error(f"Could not load the data: {e}")
    st.stop()

if df_all.empty:
    st.warning("The Fuel_Log has no usable rows (check Date, Litres and Cost are filled in).")
    st.stop()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.sidebar.subheader("📍 Filters")
df = df_all

selected_dept = st.sidebar.selectbox("Department", ["All"] + sorted(df["Department"].unique()))
if selected_dept != "All":
    df = df[df["Department"] == selected_dept]

selected_truck = st.sidebar.selectbox("Truck", ["All"] + sorted(df["Truck_Reg"].unique()))
if selected_truck != "All":
    df = df[df["Truck_Reg"] == selected_truck]

selected_supplier = st.sidebar.selectbox("Fuel Supplier", ["All"] + sorted(df["Fuel_Supplier"].unique()))
if selected_supplier != "All":
    df = df[df["Fuel_Supplier"] == selected_supplier]

min_d, max_d = df_all.index.min().date(), df_all.index.max().date()
start_date = st.sidebar.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
end_date = st.sidebar.date_input("End Date", max_d, min_value=min_d, max_value=max_d)
if start_date > end_date:
    st.sidebar.error("Start date must be on or before the end date.")
    st.stop()
df = df.loc[(df.index >= pd.Timestamp(start_date)) &
            (df.index < pd.Timestamp(end_date) + pd.Timedelta(days=1))]

if df.empty:
    st.warning("No records match the current filters.")
    st.stop()

threshold = st.sidebar.slider("Z-score threshold for anomaly", 1.0, 3.0, 2.0, 0.1)

# ---------------------------------------------------------------------------
# Shared calculations
# ---------------------------------------------------------------------------
df = add_anomaly_flags(df, threshold)
supplier_share = df.groupby("Fuel_Supplier")["Litres"].sum().reset_index()
truck_totals = df.groupby("Truck_Reg")["Litres"].sum().reset_index().sort_values("Litres", ascending=False)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Fuel Trends", "🔮 Forecasting", "📤 Export"])

with tab1:
    st.header("📊 Fuel Consumption Overview")
    st.dataframe(df.reset_index(), width="stretch")

    st.subheader("⚡ KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Litres", f"{df['Litres'].sum():,.0f}")
    c2.metric("Total Cost (KES)", f"{df['Cost'].sum():,.0f}")
    c3.metric("Avg Consumption per Truck", f"{df.groupby('Truck_Reg')['Litres'].mean().mean():,.0f} L")

    st.subheader("📍 Supplier Share")
    fig_supplier = px.pie(supplier_share, names="Fuel_Supplier", values="Litres",
                          title="Fuel Share by Supplier")
    st.plotly_chart(fig_supplier, width="stretch")

    st.subheader("🧠 Anomaly Detection")
    anomalies = df[df["Anomaly"] != ""]
    if not anomalies.empty:
        st.warning("⚠️ Consumption spikes detected (compared with the same department):")
        st.dataframe(
            anomalies.reset_index()[["Date", "Department", "Truck_Reg", "Litres",
                                     "Consumption_ZScore", "Anomaly"]],
            width="stretch",
        )
    else:
        st.success("✅ No anomalies detected.")

with tab2:
    st.header("📈 Fuel Trend Analysis")

    st.subheader("Litres Over Time by Department")
    by_dept = df.reset_index().groupby(["Date", "Department"], as_index=False)["Litres"].sum()
    st.plotly_chart(px.line(by_dept, x="Date", y="Litres", color="Department",
                            markers=True, title="Fuel Consumption by Department"),
                    width="stretch")

    st.subheader("Top Trucks by Consumption")
    st.plotly_chart(px.bar(truck_totals.head(20), x="Truck_Reg", y="Litres",
                           title="Total Fuel Consumption per Truck (top 20)"),
                    width="stretch")

    st.subheader("Supplier Comparison")
    by_supplier = df.reset_index().groupby(["Date", "Fuel_Supplier"], as_index=False)["Cost"].sum()
    st.plotly_chart(px.line(by_supplier, x="Date", y="Cost", color="Fuel_Supplier",
                            markers=True, title="Fuel Cost by Supplier"),
                    width="stretch")

with tab3:
    st.header("🔮 Forecasting Fuel Consumption and Cost per Truck")
    st.caption("Forecasts need history. Trucks with fewer periods than the minimum are skipped.")

    col_a, col_b, col_c = st.columns(3)
    freq_label = col_a.selectbox("Forecast by", ["Day", "Week", "Month"])
    freq = {"Day": "D", "Week": "W", "Month": "MS"}[freq_label]
    horizon = col_b.slider(f"{freq_label}s to forecast", 1, 30 if freq == "D" else 12, 7 if freq == "D" else 3)
    min_points = col_c.slider("Minimum periods of history", 5, 60, 14 if freq == "D" else 8)

    top_trucks = truck_totals["Truck_Reg"].head(5).tolist()
    chosen = st.multiselect("Trucks to forecast", sorted(df["Truck_Reg"].unique()), default=top_trucks)

    if st.button("Run forecast", disabled=not chosen):
        try:
            st.session_state["forecast"] = run_forecasts(df, tuple(chosen), freq, horizon, min_points)
        except ModuleNotFoundError:
            st.error("Prophet is not installed. Run: pip install prophet")

    result = st.session_state.get("forecast")
    combined_forecast = None
    if result:
        combined_forecast, too_short, failed = result
        if too_short:
            st.info(f"Skipped (not enough history): {', '.join(too_short)}")
        for msg in failed:
            st.warning(f"Forecast failed for {msg}")
        if combined_forecast is not None:
            st.subheader("📊 Forecasted Fuel Consumption and Cost per Truck")
            st.dataframe(combined_forecast, width="stretch")
            fig_forecast = px.line(combined_forecast, x="Date", y="Forecasted_Value",
                                   color="Truck_Reg", facet_row="Metric", markers=True,
                                   title="Forecasted Fuel Consumption (Litres) and Cost (KES) per Truck")
            fig_forecast.update_yaxes(matches=None)  # Litres and KES need separate scales
            st.plotly_chart(fig_forecast, width="stretch")
            st.caption("Shows the forecast from the last time you pressed Run forecast.")

with tab4:
    st.header("📤 Export Dashboard Data")
    excel_data = to_excel({
        "Fuel_Data": df.reset_index(),
        "Supplier_Share": supplier_share,
        "Truck_Totals": truck_totals,
        "Forecast_Per_Truck": combined_forecast,
    })
    st.download_button(
        label="Download Excel File",
        data=excel_data,
        file_name="fuel_dashboard_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )