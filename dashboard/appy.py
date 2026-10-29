import csv
from datetime import datetime
import os
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Soil Intelligence Digital Twin",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
DATA_DIR = os.path.join(BASE_DIR, "data")

UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PREDICTION_DIR = os.path.join(DATA_DIR, "predictions")
EXPERIMENT_DIR = os.path.join(DATA_DIR, "experiments")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
UPLOADED_GRAPH_DIR = os.path.join(GRAPH_DIR, "uploaded")

LIVE_CSV = os.path.join(CSV_DIR, "live_data.csv")

for folder_path in [
    CSV_DIR,
    UPLOAD_DIR,
    PREDICTION_DIR,
    EXPERIMENT_DIR,
    GRAPH_DIR,
    UPLOADED_GRAPH_DIR,
]:
  os.makedirs(folder_path, exist_ok=True)

HEADERS = [
    "Timestamp",
    "Experiment_ID",
    "Soil_Type",
    "Water_Added_ml",
    "Temperature_C",
    "Temperature_F",
    "Temperature_K",
    "Moisture1_Raw",
    "Moisture1_V",
    "Moisture2_Raw",
    "Moisture2_V",
    "Moisture_Avg",
    "Moisture_Diff",
    "Moisture_Spread",
    "Uniformity",
    "pH_Raw",
    "pH_Voltage",
    "Nitrogen_mg_kg",
    "Phosphorus_mg_kg",
    "Potassium_mg_kg",
    "EC_uS_cm",
]

# Guarantee live_data.csv initial schema exists
if not os.path.exists(LIVE_CSV) or os.path.getsize(LIVE_CSV) == 0:
  with open(LIVE_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADERS)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "last_prediction" not in st.session_state:
  st.session_state.last_prediction = pd.DataFrame()

if "last_prediction_name" not in st.session_state:
  st.session_state.last_prediction_name = ""

if "last_prediction_path" not in st.session_state:
  st.session_state.last_prediction_path = ""

# ============================================================
# CUSTOM CYBER-SLATE CSS (FIXED PADDING & NO CHOPPED HEADERS)
# ============================================================

st.markdown(
    """
<style>
/* 1. Prevent top header truncation */
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* 2. Fixed title with line-height preventing glyph clipping */
.section-title {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
    line-height: 1.4 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 1.2rem !important;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.main-title {
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    color: #38bdf8 !important;
    line-height: 1.3 !important;
    margin-top: 0.2rem !important;
    margin-bottom: 4px !important;
}

.main-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 14px;
    line-height: 1.5;
}

.live-badge {
    display: inline-block;
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.35);
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 16px;
}

/* 3. Dark Slate Metric Cards (Matching Node UI) */
.node-card {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}
.node-card-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94a3b8;
}
.node-card-val {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 6px 0 4px 0;
    line-height: 1.2;
}
.node-card-sub {
    font-size: 0.78rem;
    color: #64748b;
    font-family: monospace;
}

.c-cyan   { color: #38bdf8; }
.c-blue   { color: #60a5fa; }
.c-amber  { color: #fbbf24; }
.c-purple { color: #c084fc; }
.c-green  { color: #34d399; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS & STANDALONE SOFTAP RECEIVER
# ============================================================


def fetch_from_esp32_softap(timeout=0.6):
  """Polls ESP32-S3 SoftAP directly (192.168.4.1) and records to live_data.csv."""
  try:
    resp = requests.get("http://192.168.4.1/api/data", timeout=timeout)
    if resp.status_code == 200:
      d = resp.json()
      ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      temp_c = float(d.get("temp_c", 25.0))
      m1_raw = int(d.get("m1_raw", 0))
      m2_raw = int(d.get("m2_raw", 0))
      m_avg = (m1_raw + m2_raw) / 2.0
      m_diff = abs(m1_raw - m2_raw)
      unif = (1.0 - (m_diff / m_avg)) if m_avg > 0 else 1.0
      unif = max(0.0, min(1.0, unif))

      row = [
          ts,
          st.session_state.get("exp_id_val", "EXP007"),
          st.session_state.get("soil_type_val", "Oven-Dried Tested Matrix"),
          str(st.session_state.get("water_added_val", 0.0)),
          f"{temp_c:.2f}",
          f"{(temp_c * 9/5 + 32):.2f}",
          f"{(temp_c + 273.15):.2f}",
          str(m1_raw),
          f"{float(d.get('m1_v', 0.0)):.3f}",
          str(m2_raw),
          f"{float(d.get('m2_v', 0.0)):.3f}",
          f"{m_avg:.1f}",
          f"{m_diff:.1f}",
          f"{m_diff:.1f}",
          f"{unif:.3f}",
          str(d.get("ph_raw", 0)),
          f"{float(d.get('ph_v', 0.0)):.3f}",
          "0.0",
          "0.0",
          "0.0",
          f"{float(d.get('ec_comp', 0.0)):.1f}",
      ]

      with open(LIVE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
      return True
  except Exception:
    return False
  return False


def load_csv(path):
  try:
    if os.path.exists(path) and os.path.getsize(path) > 0:
      return pd.read_csv(path)
  except Exception:
    return pd.DataFrame()
  return pd.DataFrame()


def reset_live_session():
  with open(LIVE_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADERS)


def get_numeric_columns(df):
  if df.empty:
    return []
  return df.select_dtypes(include=np.number).columns.tolist()


def get_latest_value(df, column):
  if df.empty or column is None or column not in df.columns:
    return 0.0
  values = pd.to_numeric(df[column], errors="coerce").dropna()
  if values.empty:
    return 0.0
  return float(values.iloc[-1])


def get_moisture_column(df):
  for col in ["Moisture_Avg", "Moisture", "Moisture1_Raw", "Moisture2_Raw"]:
    if col in df.columns:
      return col
  return None


def get_available_datasets(live_dataframe):
  datasets = {}
  if not live_dataframe.empty:
    datasets["📡 Live Data"] = {
        "df": live_dataframe,
        "type": "Live",
        "filename": "live_data.csv",
    }

  for folder, prefix, dataset_type in [
      (UPLOAD_DIR, "📁 Uploaded", "Uploaded"),
      (PREDICTION_DIR, "🔮 Predicted", "Predicted"),
      (EXPERIMENT_DIR, "🧪 Experiment", "Experiment"),
  ]:
    if os.path.exists(folder):
      for file in sorted(os.listdir(folder)):
        if file.lower().endswith(".csv"):
          path = os.path.join(folder, file)
          loaded_df = load_csv(path)
          if not loaded_df.empty:
            datasets[f"{prefix} | {file}"] = {
                "df": loaded_df,
                "type": dataset_type,
                "filename": file,
            }
  return datasets


def get_graph_groups():
  groups = {}
  if not os.path.isdir(GRAPH_DIR):
    return groups

  for group_name in sorted(os.listdir(GRAPH_DIR)):
    group_path = os.path.join(GRAPH_DIR, group_name)
    if not os.path.isdir(group_path):
      continue

    category_files = {}
    for root, _, files in os.walk(group_path):
      image_files = sorted([
          f
          for f in files
          if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
      ])
      if image_files:
        relative_folder = os.path.relpath(root, group_path)
        category_name = (
            os.path.basename(root)
            if relative_folder != "."
            else group_name
        )
        category_files[category_name] = [
            os.path.join(root, f) for f in image_files
        ]

    if category_files:
      groups[group_name] = category_files
  return groups


def display_graph_gallery(group_name, category_files):
  st.subheader(f"📊 {group_name.replace('_', ' ')}")
  categories = list(category_files.keys())
  if not categories:
    st.info("No graphs found in this folder.")
    return

  selected_category = st.selectbox(
      "Select Graph Category", categories, key=f"graph_category_{group_name}"
  )
  image_paths = category_files[selected_category]
  st.caption(f"{len(image_paths)} graph(s) found")

  for i in range(0, len(image_paths), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
      index = i + j
      if index >= len(image_paths):
        break
      image_path = image_paths[index]
      image_name = os.path.splitext(os.path.basename(image_path))[0].replace(
          "_", " "
      )
      with col:
        st.image(image_path, use_container_width=True, caption=image_name)


def safe_extract_zip(zip_file, destination):
  os.makedirs(destination, exist_ok=True)
  destination = os.path.abspath(destination)
  with zipfile.ZipFile(zip_file, "r") as zip_ref:
    for member in zip_ref.infolist():
      member_path = os.path.abspath(os.path.join(destination, member.filename))
      if not (
          member_path == destination
          or member_path.startswith(destination + os.sep)
      ):
        continue
      if member.is_dir():
        os.makedirs(member_path, exist_ok=True)
      else:
        os.makedirs(os.path.dirname(member_path), exist_ok=True)
        with zip_ref.open(member) as source:
          with open(member_path, "wb") as target:
            shutil.copyfileobj(source, target)


def get_uploaded_graph_collections():
  if not os.path.isdir(UPLOADED_GRAPH_DIR):
    return []
  return sorted([
      name
      for name in os.listdir(UPLOADED_GRAPH_DIR)
      if os.path.isdir(os.path.join(UPLOADED_GRAPH_DIR, name))
  ])


def get_uploaded_graph_files(collection_name):
  root_folder = os.path.join(UPLOADED_GRAPH_DIR, collection_name)
  graph_files = []
  if not os.path.isdir(root_folder):
    return graph_files

  for root, _, files in os.walk(root_folder):
    for file in sorted(files):
      if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        full_path = os.path.join(root, file)
        relative_path = os.path.relpath(full_path, root_folder)
        relative_folder = os.path.dirname(relative_path)
        graph_files.append({
            "name": file,
            "path": full_path,
            "relative": relative_path,
            "folder": relative_folder if relative_folder else "Root",
        })
  return sorted(graph_files, key=lambda x: x["relative"].lower())


# ============================================================
# DIGITAL TWIN ML PREDICTOR
# ============================================================


@st.cache_resource
def load_ml_model():
  model_path = os.path.join(BASE_DIR, "models", "digital_twin_model.pkl")
  encoder_path = os.path.join(BASE_DIR, "models", "soil_type_encoder.pkl")
  try:
    import joblib

    if os.path.exists(model_path) and os.path.exists(encoder_path):
      model = joblib.load(model_path)
      encoder = joblib.load(encoder_path)
      return model, encoder
  except Exception as e:
    st.sidebar.error(f"Failed to load ML model: {e}")
  return None, None


def generate_predicted_dataset(
    base_df, water_increment, soil_type, total_water
):
  if base_df.empty:
    return pd.DataFrame()

  predicted_df = base_df.copy()
  model, encoder = load_ml_model()

  if model and encoder:
    try:
      soil_encoded = encoder.transform([soil_type])[0]
      latest_row = base_df.iloc[-1]
      current_moisture = float(latest_row.get("Moisture_Avg", 0))
      current_temp = float(latest_row.get("Temperature_C", 0))
      current_ph = float(latest_row.get("pH_Voltage", 0))

      X_transition = pd.DataFrame(
          [[
              soil_encoded,
              current_moisture,
              current_temp,
              current_ph,
              water_increment,
          ]],
          columns=[
              "Soil_Type_Encoded",
              "Current_Moisture_Avg",
              "Current_Temperature_C",
              "Current_pH_Voltage",
              "Water_Increment_ml",
          ],
      )

      pred_future = model.predict(X_transition)[0]
      new_temp, new_moist, new_ph = (
          pred_future[0],
          pred_future[1],
          pred_future[2],
      )
      delta_temp = new_temp - current_temp
      delta_moist = new_moist - current_moisture
      delta_ph = new_ph - current_ph

      if "Temperature_C" in predicted_df.columns:
        predicted_df["Temperature_C"] = (
            pd.to_numeric(predicted_df["Temperature_C"], errors="coerce")
            + delta_temp
        )
        predicted_df["Temperature_F"] = (
            predicted_df["Temperature_C"] * 9 / 5
        ) + 32
        predicted_df["Temperature_K"] = (
            predicted_df["Temperature_C"] + 273.15
        )

      for col in [
          "Moisture_Avg",
          "Moisture1_Raw",
          "Moisture2_Raw",
          "Moisture",
      ]:
        if col in predicted_df.columns:
          predicted_df[col] = (
              pd.to_numeric(predicted_df[col], errors="coerce") + delta_moist
          )

      if "pH_Voltage" in predicted_df.columns:
        predicted_df["pH_Voltage"] = (
            pd.to_numeric(predicted_df["pH_Voltage"], errors="coerce")
            + delta_ph
        )

    except Exception:
      model = None

  if not model or not encoder:
    moisture_factor = 1.2 if soil_type == "Natural Loamy" else 0.95
    temperature_factor = 0.015 if soil_type == "Natural Loamy" else 0.020
    ph_factor = 0.001 if soil_type == "Natural Loamy" else 0.0015

    for col in ["Moisture_Avg", "Moisture1_Raw", "Moisture2_Raw", "Moisture"]:
      if col in predicted_df.columns:
        predicted_df[col] = pd.to_numeric(
            predicted_df[col], errors="coerce"
        ) + (water_increment * moisture_factor)

    if "Temperature_C" in predicted_df.columns:
      predicted_df["Temperature_C"] = pd.to_numeric(
          predicted_df["Temperature_C"], errors="coerce"
      ) - (water_increment * temperature_factor)
      predicted_df["Temperature_F"] = (
          predicted_df["Temperature_C"] * 9 / 5
      ) + 32
      predicted_df["Temperature_K"] = predicted_df["Temperature_C"] + 273.15

    if "pH_Voltage" in predicted_df.columns:
      predicted_df["pH_Voltage"] = pd.to_numeric(
          predicted_df["pH_Voltage"], errors="coerce"
      ) - (water_increment * ph_factor)

  if (
      "Moisture1_Raw" in predicted_df.columns
      and "Moisture2_Raw" in predicted_df.columns
  ):
    predicted_df["Moisture_Diff"] = abs(
        predicted_df["Moisture1_Raw"] - predicted_df["Moisture2_Raw"]
    )
    predicted_df["Moisture_Spread"] = predicted_df["Moisture_Diff"]
    max_diff = predicted_df["Moisture_Diff"].max()
    if pd.notna(max_diff) and max_diff != 0:
      predicted_df["Uniformity"] = 1.0 - (
          predicted_df["Moisture_Diff"] / max_diff
      )

  predicted_df["Simulation_Water_Added_ml"] = water_increment
  predicted_df["Simulation_Total_Water_ml"] = total_water
  predicted_df["Simulation_Soil_Type"] = soil_type
  predicted_df["Prediction_Timestamp"] = datetime.now().strftime(
      "%Y-%m-%d %H:%M:%S"
  )
  return predicted_df


# ============================================================
# DATA SYNC & SIDEBAR NAVIGATION
# ============================================================

softap_active = fetch_from_esp32_softap()
live_df = load_csv(LIVE_CSV)

st.sidebar.title("🌱 Soil Intelligence")

if softap_active:
  st.sidebar.markdown(
      '<div class="live-badge">&#9679; SOFTAP CONNECTED (192.168.4.1)</div>',
      unsafe_allow_html=True,
  )
else:
  st.sidebar.caption("📡 Offline / File Polling Mode")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📡 Real-Time Monitoring",
        "📡 THz Spectroscopy Analysis",
        "📁 Upload Dataset",
        "📈 Experimental Graphs",
        "📊 Calibration Studio",
        "🔮 Digital Twin Simulator",
        "🧪 Save Experiment",
        "📂 Project Storage",
        "🗂️ Manage Datasets",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Experiment Controls")

SOIL_TYPE = st.sidebar.selectbox(
    "Soil Type",
    [
        "Oven-Dried Tested Matrix",
        "Natural Loamy",
        "<300 µm Sifted Soil",
        "<600 µm Sifted Soil",
    ],
)
st.session_state["soil_type_val"] = SOIL_TYPE

WATER_ADDED = st.sidebar.number_input(
    "Current Water Added (mL)", min_value=0.0, value=0.0, step=2.5
)
st.session_state["water_added_val"] = WATER_ADDED

EXPERIMENT_ID = st.sidebar.text_input("Experiment ID", "EXP007")
st.session_state["exp_id_val"] = EXPERIMENT_ID

# ============================================================
# 1. HOME DASHBOARD (CYBER-SLATE INTERFACE)
# ============================================================

if page == "🏠 Dashboard":
  st.markdown(
      '<div class="main-title">Multi-Modal Soil Diagnostic Node</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="main-subtitle">ESP32-S3 Autonomous Telemetry Station'
      " &bull; In-Situ Impedance & Dielectric Ground Truth</div>",
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="live-badge">&#9679; LIVE AUTONOMOUS STREAM</div>',
      unsafe_allow_html=True,
  )

  if live_df.empty or len(live_df) < 1:
    st.warning(
        "⚠️ No live telemetry detected. Connect laptop to ESP32 hotspot"
        " (SOIL_DIAGNOSTIC_NODE) or verify csv/live_data.csv."
    )
  else:
    row = live_df.iloc[-1]

    # Moisture Percentages derived from 12-bit baselines
    m1_raw = float(row.get("Moisture1_Raw", 4095))
    m2_raw = float(row.get("Moisture2_Raw", 4095))
    m1_pct = max(0.0, min(100.0, ((3475 - m1_raw) / (3475 - 1602)) * 100.0))
    m2_pct = max(0.0, min(100.0, ((3475 - m2_raw) / (3475 - 1602)) * 100.0))

    m_diff = abs(m1_raw - m2_raw)
    unif = float(row.get("Uniformity", 1.0))
    ph_v = float(row.get("pH_Voltage", 3.3))
    ph_raw = int(row.get("pH_Raw", 4095))
    calc_ph = max(0.0, min(14.0, 7.00 + ((ph_v - 1.856) * -5.714)))
    temp_c = float(row.get("Temperature_C", 29.81))
    ec_val = float(row.get("EC_uS_cm", 0.0))

    # Row 1: Moisture Channels
    c1, c2, c3 = st.columns(3)
    with c1:
      st.markdown(
          f"""
            <div class="node-card">
                <div class="node-card-title">Capacitive Moisture 1</div>
                <div class="node-card-val c-cyan">{m1_pct:.1f} %</div>
                <div class="node-card-sub">ADC: {int(m1_raw)} | Voltage: {float(row.get('Moisture1_V', 3.3)):.3f} V</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c2:
      st.markdown(
          f"""
            <div class="node-card">
                <div class="node-card-title">Capacitive Moisture 2</div>
                <div class="node-card-val c-cyan">{m2_pct:.1f} %</div>
                <div class="node-card-sub">ADC: {int(m2_raw)} | Voltage: {float(row.get('Moisture2_V', 3.3)):.3f} V</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c3:
      st.markdown(
          f"""
            <div class="node-card">
                <div class="node-card-title">Moisture Spatial Variance</div>
                <div class="node-card-val c-amber">{int(m_diff)} ADC &Delta;</div>
                <div class="node-card-sub">Uniformity Factor: {unif:.3f}</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    # Row 2: Electrochemical & Thermodynamic Channels
    c4, c5, c6 = st.columns(3)
    with c4:
      st.markdown(
          f"""
            <div class="node-card">
                <div class="node-card-title">Soil Electrochemical pH</div>
                <div class="node-card-val c-purple">{calc_ph:.2f}</div>
                <div class="node-card-sub">Voltage: {ph_v:.3f} V (ADC: {ph_raw})</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c5:
      st.markdown(
          f"""
            <div class="node-card">
                <div class="node-card-title">Soil Temperature</div>
                <div class="node-card-val c-amber">{temp_c:.2f} &deg;C</div>
                <div class="node-card-sub">DS18B20 1-Wire Digital Probe</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c6:
      st.markdown(
          f"""
            <div class="node-card">
                <div class="node-card-title">Electrical Conductivity (EC)</div>
                <div class="node-card-val c-green">{ec_val:.1f} &mu;S/cm</div>
                <div class="node-card-sub">Normalized @ 25&deg;C Reference</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.markdown("---")
    trend_cols = [
        c
        for c in [
            "Moisture1_Raw",
            "Moisture2_Raw",
            "Temperature_C",
            "pH_Voltage",
            "EC_uS_cm",
        ]
        if c in live_df.columns
    ]
    if trend_cols:
      st.subheader("📈 Live Multi-Sensor Telemetry Traces")
      st.line_chart(live_df[trend_cols].tail(60), use_container_width=True)

    st.download_button(
        "📥 Download Live Session CSV",
        live_df.to_csv(index=False).encode("utf-8"),
        file_name="live_soil_telemetry.csv",
        mime="text/csv",
        key="dashboard_dl_csv",
    )

# ============================================================
# 2. REAL-TIME MONITORING
# ============================================================

elif page == "📡 Real-Time Monitoring":
  st.markdown(
      '<div class="section-title">📡 Real-Time Telemetry Stream</div>',
      unsafe_allow_html=True,
  )

  c_ctrl1, c_ctrl2 = st.columns([1, 1])
  with c_ctrl1:
    auto_refresh = st.checkbox(
        "🔄 Continuous Hardware Stream (1s)",
        value=True,
        key="rt_stream_toggle",
    )
  with c_ctrl2:
    if st.button("🚨 Clear Live Buffer & Reinitialize Session"):
      reset_live_session()
      st.success("Telemetry buffer reinitialized to 0.")
      st.rerun()

  if live_df.empty:
    st.warning("⚠️ No records found in stream buffer.")
  else:
    row = live_df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "💧 M1 ADC",
        f"{float(row.get('Moisture1_Raw', 0)):.0f}",
        f"{float(row.get('Moisture1_V', 0)):.2f} V",
    )
    c2.metric(
        "💧 M2 ADC",
        f"{float(row.get('Moisture2_Raw', 0)):.0f}",
        f"{float(row.get('Moisture2_V', 0)):.2f} V",
    )
    c3.metric(
        "🌡 Temperature", f"{float(row.get('Temperature_C', 25)):.2f} °C"
    )
    c4.metric("🧪 pH Probe V", f"{float(row.get('pH_Voltage', 0)):.3f} V")
    c5.metric("⚡ EC (25°C)", f"{float(row.get('EC_uS_cm', 0)):.1f} µS/cm")

    st.subheader("📊 Dynamic Data Table (Last 30 Pulses)")
    st.dataframe(live_df.tail(30), use_container_width=True)

    num_cols = get_numeric_columns(live_df)
    selected_cols = st.multiselect(
        "Select Channels to Plot",
        num_cols,
        default=[
            c
            for c in ["Moisture1_Raw", "Moisture2_Raw", "EC_uS_cm"]
            if c in num_cols
        ],
    )
    if selected_cols:
      fig = px.line(
          live_df.tail(150),
          y=selected_cols,
          title="Live High-Frequency Sensor Transduction",
      )
      fig.update_layout(
          plot_bgcolor="white",
          xaxis=dict(gridcolor="#eeeeee"),
          yaxis=dict(gridcolor="#eeeeee"),
      )
      st.plotly_chart(fig, use_container_width=True)

  if auto_refresh:
    time.sleep(1.0)
    st.rerun()

# ============================================================
# 3. UPLOAD DATASET
# ============================================================

elif page == "📁 Upload Dataset":
  st.markdown(
      '<div class="section-title">📁 Upload Laboratory Datasets</div>',
      unsafe_allow_html=True,
  )

  uploaded_file = st.file_uploader(
      "Upload Laboratory CSV Dataset", type=["csv"], key="lab_dataset_upload"
  )
  if uploaded_file:
    try:
      upload_df = pd.read_csv(uploaded_file)
      file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
      upload_df.to_csv(file_path, index=False)
      st.success(f"✅ Stored `{uploaded_file.name}` under `{UPLOAD_DIR}`.")
      st.dataframe(upload_df.head(25), use_container_width=True)
    except Exception as e:
      st.error(f"Error parsing CSV: {e}")

  st.markdown("---")
  st.subheader("📂 Uploaded Datasets Archive")
  uploaded_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]
  if uploaded_files:
    for f_name in uploaded_files:
      cf, cd = st.columns([3, 1])
      cf.write(f"📄 **{f_name}**")
      d_df = load_csv(os.path.join(UPLOAD_DIR, f_name))
      cd.download_button(
          "📥 Download",
          d_df.to_csv(index=False).encode("utf-8"),
          file_name=f_name,
          key=f"dl_stored_{f_name}",
      )
  else:
    st.info("No external datasets stored.")

# ============================================================
# 4. EXPERIMENTAL GRAPHS
# ============================================================

elif page == "📈 Experimental Graphs":
  st.markdown(
      '<div class="section-title">📈 Experimental Visual Analytics</div>',
      unsafe_allow_html=True,
  )

  st.subheader("📦 Upload Graph Archive (.ZIP)")
  uploaded_zips = st.file_uploader(
      "Upload ZIP collections containing high-resolution plots",
      type=["zip"],
      accept_multiple_files=True,
  )

  if uploaded_zips:
    for u_zip in uploaded_zips:
      z_name = os.path.splitext(u_zip.name)[0]
      dest = os.path.join(UPLOADED_GRAPH_DIR, z_name)
      marker = os.path.join(dest, ".extracted")

      if not os.path.exists(marker):
        try:
          if os.path.exists(dest):
            shutil.rmtree(dest)
          os.makedirs(dest, exist_ok=True)
          safe_extract_zip(u_zip, dest)
          with open(marker, "w") as m:
            m.write(u_zip.name)
          st.success(f"✅ Unpacked `{u_zip.name}` successfully.")
        except Exception as e:
          st.error(f"Extraction error: {e}")

  st.markdown("---")
  collections = get_uploaded_graph_collections()
  if collections:
    st.subheader("📂 Active Graph Collections")
    sel_col = st.selectbox(
        "Select Uploaded ZIP Container", collections, key="sel_col_zip"
    )
    g_files = get_uploaded_graph_files(sel_col)

    if not g_files:
      st.warning("No image files (.png, .jpg) discovered inside archive.")
    else:
      st.success(f"Discovered {len(g_files)} graph(s) inside `{sel_col}`.")
      folders = sorted(list(set(g["folder"] for g in g_files)))
      sel_folder = st.selectbox("Filter Category / Subdirectory", ["All"] + folders)

      filt_graphs = (
          g_files
          if sel_folder == "All"
          else [g for g in g_files if g["folder"] == sel_folder]
      )

      search_kw = st.text_input("🔎 Search Graph Filename", "")
      if search_kw.strip():
        filt_graphs = [
            g
            for g in filt_graphs
            if search_kw.lower() in g["name"].lower()
            or search_kw.lower() in g["relative"].lower()
        ]

      st.write(f"Displaying **{len(filt_graphs)}** figures.")
      if filt_graphs:
        opts = [g["relative"] for g in filt_graphs]
        sel_fig = st.selectbox("Inspect Individual Plot", opts)
        chosen = next(g for g in filt_graphs if g["relative"] == sel_fig)

        st.subheader(f"📈 {chosen['name']}")
        st.caption(f"Path: {chosen['relative']}")
        st.image(chosen["path"], use_container_width=True)

        with open(chosen["path"], "rb") as bf:
          b_data = bf.read()
        st.download_button(
            "⬇️ Download High-Res Image",
            b_data,
            file_name=chosen["name"],
            key=f"dl_single_img_{chosen['name']}",
        )

  st.markdown("---")
  st.subheader("📊 Existing System Graphs")
  sys_groups = {
      k: v for k, v in get_graph_groups().items() if k != "uploaded"
  }
  if sys_groups:
    chosen_grp = st.selectbox(
        "Select Native Folder", list(sys_groups.keys()), key="native_grp_sel"
    )
    display_graph_gallery(chosen_grp, sys_groups[chosen_grp])
  else:
    st.info("No pre-packaged graph folders found.")

# ============================================================
# 5. CALIBRATION STUDIO
# ============================================================

elif page == "📊 Calibration Studio":
  st.markdown(
      '<div class="section-title">📊 Multi-Sensor Calibration Studio</div>',
      unsafe_allow_html=True,
  )
  st.caption(
      "Decouple empirical sensor transduction across water gradients ($0.0$ to"
      " $10.0$ mL)."
  )

  datasets = get_available_datasets(live_df)
  if not datasets:
    st.warning("⚠️ No calibration matrices located. Upload data to begin.")
  else:
    d_names = list(datasets.keys())
    sel_datasets = st.multiselect(
        "Select Comparison Matrices", d_names, default=[d_names[0]]
    )

    if sel_datasets:
      fig_m_adc = go.Figure()
      fig_m_pct = go.Figure()
      fig_ec = go.Figure()
      fig_temp = go.Figure()
      fig_ph = go.Figure()

      palette = [
          "#38bdf8",
          "#f59e0b",
          "#34d399",
          "#c084fc",
          "#f43f5e",
          "#60a5fa",
      ]
      layout_opts = dict(
          plot_bgcolor="white",
          xaxis=dict(
              showgrid=True,
              gridcolor="#f1f5f9",
              title="Water Volume Added (mL)",
          ),
          yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
          legend=dict(orientation="h", y=1.05, x=0),
          margin=dict(l=50, r=30, t=50, b=50),
      )

      for idx, name in enumerate(sel_datasets):
        c = palette[idx % len(palette)]
        df = datasets[name]["df"].copy()

        if "Water_Added_ml" in df.columns:
          df = df.sort_values("Water_Added_ml")
          x = df["Water_Added_ml"]

          if "Moisture_Avg" in df.columns:
            y_m = pd.to_numeric(df["Moisture_Avg"], errors="coerce")
            fig_m_adc.add_trace(
                go.Scatter(
                    x=x,
                    y=y_m,
                    mode="lines+markers",
                    name=f"{name} (ADC)",
                    line=dict(color=c, width=2.5),
                    marker=dict(size=8),
                )
            )

            # Volumetric percentage conversion
            y_pct = ((3475 - y_m) / (3475 - 1602)) * 100.0
            fig_m_pct.add_trace(
                go.Scatter(
                    x=x,
                    y=y_pct.clip(0, 100),
                    mode="lines+markers",
                    name=f"{name} (%)",
                    line=dict(color=c, width=2.5),
                    marker=dict(size=8, symbol="square"),
                )
            )

          if "EC_uS_cm" in df.columns:
            fig_ec.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(df["EC_uS_cm"], errors="coerce"),
                    mode="lines+markers",
                    name=f"{name} (EC)",
                    line=dict(color=c, width=2.5),
                    marker=dict(size=8, symbol="triangle-up"),
                )
            )

          if "Temperature_C" in df.columns:
            fig_temp.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(df["Temperature_C"], errors="coerce"),
                    mode="lines+markers",
                    name=f"{name} (°C)",
                    line=dict(color=c, width=2.5),
                    marker=dict(size=8),
                )
            )

          if "pH_Voltage" in df.columns:
            fig_ph.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(df["pH_Voltage"], errors="coerce"),
                    mode="lines+markers",
                    name=f"{name} (pH V)",
                    line=dict(color=c, width=2.5),
                    marker=dict(size=8, symbol="diamond"),
                )
            )

      fig_m_adc.update_layout(
          title="<b>1. Capacitive Transduction: 12-bit ADC vs Water Added</b>",
          yaxis_title="ADC Output Register",
          **layout_opts,
      )
      st.plotly_chart(fig_m_adc, use_container_width=True)

      fig_m_pct.update_layout(
          title="<b>2. Calibrated Moisture Content (%) vs Water Added</b>",
          yaxis_title="Calculated Moisture (%)",
          **layout_opts,
      )
      st.plotly_chart(fig_m_pct, use_container_width=True)

      fig_ec.update_layout(
          title="<b>3. Bulk Electrical Conductivity (EC @ 25°C)</b>",
          yaxis_title="Conductivity (µS/cm)",
          **layout_opts,
      )
      st.plotly_chart(fig_ec, use_container_width=True)

      fig_temp.update_layout(
          title="<b>4. Soil Thermodynamic Equilibrium (°C)</b>",
          yaxis_title="Temperature (°C)",
          **layout_opts,
      )
      st.plotly_chart(fig_temp, use_container_width=True)

      fig_ph.update_layout(
          title="<b>5. Electrochemical Cell Potential (pH Sensor Volts)</b>",
          yaxis_title="Output Potential (V)",
          **layout_opts,
      )
      st.plotly_chart(fig_ph, use_container_width=True)

# ============================================================
# 6. DIGITAL TWIN SIMULATOR
# ============================================================

elif page == "🔮 Digital Twin Simulator":
  st.markdown(
      '<div class="section-title">🔮 Digital Twin State Transition'
      " Simulator</div>",
      unsafe_allow_html=True,
  )

  datasets = get_available_datasets(live_df)
  if not datasets:
    st.warning("⚠️ No baseline matrix available for simulation.")
  else:
    b_name = st.selectbox(
        "Select Baseline Ground Truth Matrix", list(datasets.keys())
    )
    b_df = datasets[b_name]["df"].copy()

    m_col = get_moisture_column(b_df)
    cur_m = get_latest_value(b_df, m_col)
    cur_t = get_latest_value(b_df, "Temperature_C")
    cur_p = get_latest_value(b_df, "pH_Voltage")
    cur_u = get_latest_value(b_df, "Uniformity")

    st.subheader("Current Base States")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("💧 Base Moisture", f"{cur_m:.0f} ADC")
    s2.metric("🌡 Base Temp", f"{cur_t:.2f} °C")
    s3.metric("🧪 Base pH Pot.", f"{cur_p:.3f} V")
    s4.metric("📊 Uniformity", f"{cur_u:.3f}")

    water_inc = st.number_input(
        "Simulate Injected Water (mL)", min_value=0.0, value=2.5, step=2.5
    )
    tot_water = WATER_ADDED + water_inc

    if st.button(
        "🔮 Compute Physical State Transition",
        type="primary",
        use_container_width=True,
    ):
      pred_df = generate_predicted_dataset(
          b_df, water_inc, SOIL_TYPE, tot_water
      )
      if not pred_df.empty:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_name = (
            f"{EXPERIMENT_ID}_PREDICTED_{int(tot_water)}ml_{ts_tag}.csv"
        )
        sim_path = os.path.join(PREDICTION_DIR, sim_name)
        pred_df.to_csv(sim_path, index=False)

        st.session_state.last_prediction = pred_df
        st.session_state.last_prediction_name = sim_name
        st.success("✅ State transition computed and saved!")

    if not st.session_state.last_prediction.empty:
      p_df = st.session_state.last_prediction
      p_m = get_latest_value(p_df, get_moisture_column(p_df))
      p_t = get_latest_value(p_df, "Temperature_C")
      p_p = get_latest_value(p_df, "pH_Voltage")
      p_u = get_latest_value(p_df, "Uniformity")

      st.markdown("---")
      st.subheader("Simulated Transition Output")
      r1, r2, r3, r4 = st.columns(4)
      r1.metric(
          "💧 Simulated Moisture",
          f"{p_m:.0f} ADC",
          delta=f"{p_m - cur_m:.0f} ADC",
      )
      r2.metric(
          "🌡 Simulated Temp",
          f"{p_t:.2f} °C",
          delta=f"{p_t - cur_t:.2f} °C",
      )
      r3.metric(
          "🧪 Simulated pH",
          f"{p_p:.3f} V",
          delta=f"{p_p - cur_p:.3f} V",
      )
      r4.metric("📊 Simulated Uniformity", f"{p_u:.3f}")

      st.dataframe(p_df.tail(15), use_container_width=True)
      st.download_button(
          "📥 Download Simulated CSV",
          p_df.to_csv(index=False).encode("utf-8"),
          file_name=st.session_state.last_prediction_name,
          mime="text/csv",
      )

# ============================================================
# 7. SAVE EXPERIMENT
# ============================================================

elif page == "🧪 Save Experiment":
  st.markdown(
      '<div class="section-title">🧪 Freeze In-Situ Experiment Snapshot</div>',
      unsafe_allow_html=True,
  )

  if live_df.empty:
    st.warning("⚠️ Telemetry stream empty. No active data to freeze.")
  else:
    st.write(
        "Locks current live stream into a permanent reference CSV under"
        " `data/experiments/`."
    )
    if st.button("💾 Freeze & Commit Active Session", type="primary"):
      ts = datetime.now().strftime("%Y%m%d_%H%M%S")
      fname = (
          f"{EXPERIMENT_ID}_ACTUAL_{int(WATER_ADDED)}ml_{ts}.csv"
      )
      fpath = os.path.join(EXPERIMENT_DIR, fname)
      live_df.to_csv(fpath, index=False)
      st.success(f"✅ Frozen into `{fname}` successfully!")
      st.download_button(
          "📥 Download Experiment File",
          live_df.to_csv(index=False).encode("utf-8"),
          file_name=fname,
          mime="text/csv",
      )

# ============================================================
# 8. PROJECT STORAGE
# ============================================================

elif page == "📂 Project Storage":
  st.markdown(
      '<div class="section-title">📂 Project Repository & Archive</div>',
      unsafe_allow_html=True,
  )

  cats = {
      "📁 Uploaded Datasets": UPLOAD_DIR,
      "🔮 Predicted Datasets": PREDICTION_DIR,
      "🧪 Experiment Datasets": EXPERIMENT_DIR,
  }

  summary = [
      {"Category": k, "Files": len([f for f in os.listdir(v) if f.endswith(".csv")])}
      for k, v in cats.items()
  ]
  st.table(pd.DataFrame(summary))

  for cat_name, folder in cats.items():
    st.markdown(f"#### {cat_name}")
    flist = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if not flist:
      st.caption("No records stored.")
    else:
      for f in flist:
        col_lbl, col_btn = st.columns([3, 1])
        col_lbl.write(f"📄 `{f}`")
        item_df = load_csv(os.path.join(folder, f))
        col_btn.download_button(
            "📥 Download",
            item_df.to_csv(index=False).encode("utf-8"),
            file_name=f,
            key=f"rep_{cat_name}_{f}",
        )

# ============================================================
# 9. MANAGE DATASETS
# ============================================================

elif page == "🗂️ Manage Datasets":
  st.markdown(
      '<div class="section-title">🗂️ Dataset Maintenance & Pruning</div>',
      unsafe_allow_html=True,
  )

  cat_map = {
      "📁 Uploaded Datasets": UPLOAD_DIR,
      "🔮 Predicted Datasets": PREDICTION_DIR,
      "🧪 Experiment Datasets": EXPERIMENT_DIR,
  }
  chosen_cat = st.selectbox("Select Target Directory", list(cat_map.keys()))
  target_dir = cat_map[chosen_cat]
  csv_files = sorted([f for f in os.listdir(target_dir) if f.endswith(".csv")])

  if not csv_files:
    st.warning("No files present.")
  else:
    chosen_file = st.selectbox("Inspect File", csv_files)
    prev_df = load_csv(os.path.join(target_dir, chosen_file))
    st.dataframe(prev_df.head(40), use_container_width=True)

    st.markdown("---")
    to_delete = st.multiselect("Select Files for Permanent Removal", csv_files)
    confirm = st.checkbox("Confirm Deletion of Selected Records")

    if st.button("🗑️ Delete Selected Datasets") and to_delete and confirm:
      for del_f in to_delete:
        del_p = os.path.join(target_dir, del_f)
        if os.path.exists(del_p):
          os.remove(del_p)
      st.success("Files successfully pruned from repository.")
      st.rerun()

# ============================================================
# 10. THz SPECTROSCOPY ANALYSIS (FULL FRESNEL INVERSION)
# ============================================================

elif page == "📡 THz Spectroscopy Analysis":
  st.markdown(
      '<div class="section-title">📡 THz-TDS Optical Inversion Engine</div>',
      unsafe_allow_html=True,
  )
  st.caption(
      "Fast Fourier Transform (FFT) Decoupling & Complex Fresnel Extraction"
      " ($0.25$ to $1.20$ THz)"
  )

  thz_uploads = st.file_uploader(
      "Upload Time-Domain CSV Scans (Include Reference `4mm SH.csv` / `8mm"
      " SH.csv`)",
      type=["csv"],
      accept_multiple_files=True,
  )

  if thz_uploads:
    dfs = {}
    for up_f in thz_uploads:
      try:
        t_df = pd.read_csv(up_f, skiprows=[1])
        if "time" not in t_df.columns or "amplitude" not in t_df.columns:
          up_f.seek(0)
          t_df = pd.read_csv(up_f)

        if "time" in t_df.columns and "amplitude" in t_df.columns:
          t_df = (
              t_df.dropna(subset=["time", "amplitude"])
              .sort_values("time")
              .reset_index(drop=True)
          )
          dfs[up_f.name] = t_df
      except Exception as err:
        st.error(f"Error loading {up_f.name}: {err}")

    if dfs:
      colors = [
          "#38bdf8",
          "#f59e0b",
          "#34d399",
          "#c084fc",
          "#f43f5e",
          "#60a5fa",
      ]
      layout_opts = dict(
          plot_bgcolor="white",
          xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
          yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
          margin=dict(l=50, r=30, t=50, b=50),
          legend=dict(orientation="h", y=1.05, x=0),
      )

      # Auto-detect empty cuvette reference
      ref_key = next(
          (
              k
              for k in dfs.keys()
              if "SH.csv" in k or "empty" in k.lower() or "sh" in k.lower()
          ),
          None,
      )

      st.subheader("1. Time-Domain Terahertz Electric Field Pulses")
      fig_time = go.Figure()
      for idx, (s_name, s_df) in enumerate(dfs.items()):
        c = colors[idx % len(colors)]
        fig_time.add_trace(
            go.Scatter(
                x=s_df["time"],
                y=s_df["amplitude"],
                mode="lines",
                name=s_name,
                line=dict(color=c, width=2.0),
            )
        )

      fig_time.update_layout(
          title="<b>Transmitted Pulse Waveforms E(t)</b>",
          xaxis_title="Delay Time (ps)",
          yaxis_title="Normalized Amplitude (a.u.)",
          **layout_opts,
      )
      st.plotly_chart(fig_time, use_container_width=True)

      if ref_key:
        st.success(
            f"✅ Reference Identified: `{ref_key}`. Computing Fresnel transfer"
            " function..."
        )

        # Uniform interpolation grid across active acquisition window
        t_grid = np.linspace(90.0, 118.0, 2048)
        dt = (t_grid[1] - t_grid[0]) * 1e-12
        n_fft = 4096

        freq = np.fft.rfftfreq(n_fft, d=dt)
        freq_thz = freq / 1e12
        omega = 2.0 * np.pi * freq
        omega_safe = np.where(omega == 0, 1e-12, omega)
        valid_mask = (freq_thz >= 0.25) & (freq_thz <= 1.20)
        c_speed = 2.99792458e8

        # Cuvette optical path length (m)
        d_cuvette = 8.0e-3 if "8mm" in ref_key else 4.0e-3

        df_ref = dfs[ref_key]
        amp_ref = np.interp(
            t_grid, df_ref["time"].values, df_ref["amplitude"].values
        )
        F_ref = np.fft.rfft(amp_ref - np.mean(amp_ref[:80]), n=n_fft)

        fig_n = go.Figure()
        fig_alpha = go.Figure()
        fig_eps = go.Figure()
        f_sub = freq_thz[valid_mask]

        for idx, (s_name, s_df) in enumerate(dfs.items()):
          if s_name == ref_key:
            continue
          c = colors[idx % len(colors)]

          amp_s = np.interp(
              t_grid, s_df["time"].values, s_df["amplitude"].values
          )
          F_s = np.fft.rfft(amp_s - np.mean(amp_s[:80]), n=n_fft)

          # Transfer Function H(w)
          H = F_s / np.where(np.abs(F_ref) == 0, 1e-10, F_ref)
          phase = np.unwrap(np.angle(H))
          amp_ratio = np.abs(H)

          # Fresnel Decoupled Optical Constants
          n_f = 1.0 - (c_speed / (omega_safe * d_cuvette)) * phase
          alpha_f = -(2.0 / (d_cuvette * 100.0)) * np.log(
              np.clip(amp_ratio, 1e-5, 1.0)
          )
          kappa_f = (alpha_f * 100.0 * c_speed) / (2.0 * omega_safe)
          eps_real = n_f**2 - kappa_f**2

          fig_n.add_trace(
              go.Scatter(
                  x=f_sub,
                  y=n_f[valid_mask],
                  mode="lines",
                  name=f"{s_name} (n)",
                  line=dict(color=c, width=2.0),
              )
          )
          fig_alpha.add_trace(
              go.Scatter(
                  x=f_sub,
                  y=alpha_f[valid_mask],
                  mode="lines",
                  name=f"{s_name} (α)",
                  line=dict(color=c, width=2.0),
              )
          )
          fig_eps.add_trace(
              go.Scatter(
                  x=f_sub,
                  y=eps_real[valid_mask],
                  mode="lines",
                  name=f"{s_name} (ε')",
                  line=dict(color=c, width=2.0),
              )
          )

        fig_n.update_layout(
            title="<b>2. Extracted Refractive Index n(f)</b>",
            xaxis_title="Frequency (THz)",
            yaxis_title="Refractive Index n",
            **layout_opts,
        )
        st.plotly_chart(fig_n, use_container_width=True)

        fig_alpha.update_layout(
            title="<b>3. Beer-Lambert Absorption Coefficient α(f)</b>",
            xaxis_title="Frequency (THz)",
            yaxis_title="Absorption Coefficient (cm⁻¹)",
            **layout_opts,
        )
        st.plotly_chart(fig_alpha, use_container_width=True)

        fig_eps.update_layout(
            title=(
                "<b>4. Real Dielectric Permittivity ε'(f) (Optical Ground"
                " Truth)</b>"
            ),
            xaxis_title="Frequency (THz)",
            yaxis_title="Real Permittivity ε'",
            **layout_opts,
        )
        st.plotly_chart(fig_eps, use_container_width=True)
      else:
        st.warning(
            "Upload reference scan (`4mm SH.csv` or `8mm SH.csv`) to unlock"
            " optical parameter inversion."
        )