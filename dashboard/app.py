import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import csv
import zipfile
import shutil

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Soil Intelligence Digital Twin",
    page_icon="🌱",
    layout="wide"
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

# NEW: Separate folder for uploaded graph ZIP files
UPLOADED_GRAPH_DIR = os.path.join(GRAPH_DIR, "uploaded")

LIVE_CSV = os.path.join(CSV_DIR, "live_data.csv")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(UPLOADED_GRAPH_DIR, exist_ok=True)

HEADERS = [
    "Timestamp", "Experiment_ID", "Soil_Type", "Water_Added_ml",
    "Temperature_C", "Temperature_F", "Temperature_K",
    "Moisture1_Raw", "Moisture1_V", "Moisture2_Raw", "Moisture2_V",
    "Moisture_Avg", "Moisture_Diff", "Moisture_Spread", "Uniformity",
    "pH_Raw", "pH_Voltage",
    "Nitrogen_mg_kg", "Phosphorus_mg_kg", "Potassium_mg_kg", "EC_uS_cm"
]

# ============================================================
# SESSION STATE
# ============================================================

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = pd.DataFrame()

if "last_prediction_name" not in st.session_state:
    st.session_state.last_prediction_name = ""

if "last_prediction_path" not in st.session_state:
    st.session_state.last_prediction_path = ""

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    font-family: 'Inter', sans-serif;
}
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #2e7d32, #81c784);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.section-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2e7d32;
    margin-bottom: 20px;
    border-bottom: 2px solid #e8f5e9;
    padding-bottom: 10px;
}
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] {
    color: #555555 !important;
}
[data-testid="stMetricValue"] {
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def reset_live_session():
    """Wipes live_data.csv and reinitializes with headers to start a new session."""
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
    possible_columns = [
        "Moisture_Avg",
        "Moisture",
        "Moisture1_Raw",
        "Moisture2_Raw"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


def get_available_datasets(live_df):
    datasets = {}

    if not live_df.empty:
        datasets["📡 Live Data"] = {
            "df": live_df,
            "type": "Live",
            "filename": "live_data.csv"
        }

    for folder, prefix, dataset_type in [
        (UPLOAD_DIR, "📁 Uploaded", "Uploaded"),
        (PREDICTION_DIR, "🔮 Predicted", "Predicted"),
        (EXPERIMENT_DIR, "🧪 Experiment", "Experiment")
    ]:
        if os.path.exists(folder):
            for file in sorted(os.listdir(folder)):
                if file.lower().endswith(".csv"):
                    path = os.path.join(folder, file)
                    df = load_csv(path)

                    if not df.empty:
                        datasets[f"{prefix} | {file}"] = {
                            "df": df,
                            "type": dataset_type,
                            "filename": file
                        }

    return datasets


# ============================================================
# EXISTING GRAPH FUNCTION
# ============================================================

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
                f for f in files
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".webp")
                )
            ])

            if image_files:

                relative_folder = os.path.relpath(
                    root,
                    group_path
                )

                category_name = (
                    os.path.basename(root)
                    if relative_folder != "."
                    else group_name
                )

                category_files[category_name] = [
                    os.path.join(root, f)
                    for f in image_files
                ]

        if category_files:
            groups[group_name] = category_files

    return groups


def display_graph_gallery(group_name, category_files):

    st.subheader(
        f"📊 {group_name.replace('_', ' ')}"
    )

    categories = list(category_files.keys())

    if not categories:
        st.info("No graphs found in this folder.")
        return

    selected_category = st.selectbox(
        "Select Graph Category",
        categories,
        key=f"graph_category_{group_name}"
    )

    image_paths = category_files[selected_category]

    st.caption(
        f"{len(image_paths)} graph(s) found"
    )

    for i in range(0, len(image_paths), 2):

        cols = st.columns(2)

        for j, col in enumerate(cols):

            index = i + j

            if index >= len(image_paths):
                break

            image_path = image_paths[index]

            image_name = os.path.splitext(
                os.path.basename(image_path)
            )[0].replace("_", " ")

            with col:

                st.image(
                    image_path,
                    use_container_width=True,
                    caption=image_name
                )


# ============================================================
# NEW ZIP GRAPH FUNCTIONS
# ============================================================

def safe_extract_zip(zip_file, destination):

    """
    Extracts a ZIP safely while preventing path traversal.
    """

    os.makedirs(destination, exist_ok=True)

    destination = os.path.abspath(destination)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:

        for member in zip_ref.infolist():

            member_path = os.path.abspath(
                os.path.join(
                    destination,
                    member.filename
                )
            )

            if not (
                member_path == destination
                or member_path.startswith(
                    destination + os.sep
                )
            ):
                continue

            if member.is_dir():

                os.makedirs(
                    member_path,
                    exist_ok=True
                )

            else:

                os.makedirs(
                    os.path.dirname(member_path),
                    exist_ok=True
                )

                with zip_ref.open(member) as source:
                    with open(member_path, "wb") as target:
                        shutil.copyfileobj(
                            source,
                            target
                        )


def get_uploaded_graph_collections():

    if not os.path.isdir(UPLOADED_GRAPH_DIR):
        return []

    return sorted([
        name
        for name in os.listdir(UPLOADED_GRAPH_DIR)
        if os.path.isdir(
            os.path.join(
                UPLOADED_GRAPH_DIR,
                name
            )
        )
    ])


def get_uploaded_graph_files(collection_name):

    root_folder = os.path.join(
        UPLOADED_GRAPH_DIR,
        collection_name
    )

    graph_files = []

    if not os.path.isdir(root_folder):
        return graph_files

    for root, _, files in os.walk(root_folder):

        for file in sorted(files):

            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ):

                full_path = os.path.join(
                    root,
                    file
                )

                relative_path = os.path.relpath(
                    full_path,
                    root_folder
                )

                relative_folder = os.path.dirname(
                    relative_path
                )

                graph_files.append({
                    "name": file,
                    "path": full_path,
                    "relative": relative_path,
                    "folder": (
                        relative_folder
                        if relative_folder
                        else "Root"
                    )
                })

    return sorted(
        graph_files,
        key=lambda x: x["relative"].lower()
    )


# ============================================================
# DIGITAL TWIN PREDICTION
# ============================================================

@st.cache_resource
def load_ml_model():

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "digital_twin_model.pkl"
    )

    encoder_path = os.path.join(
        BASE_DIR,
        "models",
        "soil_type_encoder.pkl"
    )

    try:

        import joblib

        if os.path.exists(model_path) and os.path.exists(encoder_path):

            model = joblib.load(model_path)
            encoder = joblib.load(encoder_path)

            return model, encoder

    except Exception as e:

        st.sidebar.error(
            f"Failed to load ML model: {e}"
        )

    return None, None

@st.cache_resource
def load_cross_modal_models():
    models_dir = os.path.join(BASE_DIR, "models")
    try:
        import joblib, json
        clf_p = os.path.join(models_dir, "regime_classifier.pkl")
        le_p = os.path.join(models_dir, "regime_label_encoder.pkl")
        w_p = os.path.join(models_dir, "water_regressor.pkl")
        l_p = os.path.join(models_dir, "ldpe_regressor.pkl")
        thz_p = os.path.join(models_dir, "virtual_thz_model.pkl")
        meta_p = os.path.join(models_dir, "virtual_thz_metadata.json")
        
        if os.path.exists(clf_p) and os.path.exists(thz_p) and os.path.exists(meta_p):
            clf = joblib.load(clf_p)
            le = joblib.load(le_p)
            reg_w = joblib.load(w_p)
            reg_l = joblib.load(l_p)
            reg_thz = joblib.load(thz_p)
            meta = json.load(open(meta_p, "r", encoding="utf-8"))
            return {
                "clf": clf, "le": le, "reg_w": reg_w, "reg_l": reg_l, "reg_thz": reg_thz, "meta": meta
            }
    except Exception as e:
        st.sidebar.error(f"Cross-modal model loader error: {e}")
    return None


def generate_predicted_dataset(
    base_df,
    water_increment,
    soil_type,
    total_water
):

    if base_df.empty:
        return pd.DataFrame()

    predicted_df = base_df.copy()

    model, encoder = load_ml_model()

    if model and encoder:

        try:

            soil_encoded = encoder.transform(
                [soil_type]
            )[0]

            latest_row = base_df.iloc[-1]

            current_moisture = float(
                latest_row.get(
                    "Moisture_Avg",
                    0
                )
            )

            current_temp = float(
                latest_row.get(
                    "Temperature_C",
                    0
                )
            )

            current_ph = float(
                latest_row.get(
                    "pH_Voltage",
                    0
                )
            )

            X_transition = pd.DataFrame(
                [[
                    soil_encoded,
                    current_moisture,
                    current_temp,
                    current_ph,
                    water_increment
                ]],
                columns=[
                    "Soil_Type_Encoded",
                    "Current_Moisture_Avg",
                    "Current_Temperature_C",
                    "Current_pH_Voltage",
                    "Water_Increment_ml"
                ]
            )

            pred_future = model.predict(
                X_transition
            )[0]

            new_temp = pred_future[0]
            new_moist = pred_future[1]
            new_ph = pred_future[2]

            delta_temp = new_temp - current_temp
            delta_moist = new_moist - current_moisture
            delta_ph = new_ph - current_ph

            if "Temperature_C" in predicted_df.columns:

                predicted_df["Temperature_C"] = (
                    pd.to_numeric(
                        predicted_df["Temperature_C"],
                        errors="coerce"
                    )
                    + delta_temp
                )

                predicted_df["Temperature_F"] = (
                    predicted_df["Temperature_C"]
                    * 9 / 5
                ) + 32

                predicted_df["Temperature_K"] = (
                    predicted_df["Temperature_C"]
                    + 273.15
                )

            moisture_columns = [
                "Moisture_Avg",
                "Moisture1_Raw",
                "Moisture2_Raw",
                "Moisture"
            ]

            for col in moisture_columns:

                if col in predicted_df.columns:

                    predicted_df[col] = (
                        pd.to_numeric(
                            predicted_df[col],
                            errors="coerce"
                        )
                        + delta_moist
                    )

            if "pH_Voltage" in predicted_df.columns:

                predicted_df["pH_Voltage"] = (
                    pd.to_numeric(
                        predicted_df["pH_Voltage"],
                        errors="coerce"
                    )
                    + delta_ph
                )

        except Exception as e:

            st.error(
                f"ML Prediction Error: {e}"
            )

            model = None

    if not model or not encoder:

        moisture_factor = (
            1.2
            if soil_type == "Natural Loamy"
            else 0.9
        )

        temperature_factor = (
            0.015
            if soil_type == "Natural Loamy"
            else 0.020
        )

        ph_factor = (
            0.001
            if soil_type == "Natural Loamy"
            else 0.0015
        )

        moisture_columns = [
            "Moisture_Avg",
            "Moisture1_Raw",
            "Moisture2_Raw",
            "Moisture"
        ]

        for col in moisture_columns:

            if col in predicted_df.columns:

                predicted_df[col] = (
                    pd.to_numeric(
                        predicted_df[col],
                        errors="coerce"
                    )
                    + (
                        water_increment
                        * moisture_factor
                    )
                )

        if "Temperature_C" in predicted_df.columns:

            predicted_df["Temperature_C"] = (
                pd.to_numeric(
                    predicted_df["Temperature_C"],
                    errors="coerce"
                )
                - (
                    water_increment
                    * temperature_factor
                )
            )

            predicted_df["Temperature_F"] = (
                predicted_df["Temperature_C"]
                * 9 / 5
            ) + 32

            predicted_df["Temperature_K"] = (
                predicted_df["Temperature_C"]
                + 273.15
            )

        if "pH_Voltage" in predicted_df.columns:

            predicted_df["pH_Voltage"] = (
                pd.to_numeric(
                    predicted_df["pH_Voltage"],
                    errors="coerce"
                )
                - (
                    water_increment
                    * ph_factor
                )
            )

    if (
        "Moisture1_Raw" in predicted_df.columns
        and
        "Moisture2_Raw" in predicted_df.columns
    ):

        predicted_df["Moisture_Diff"] = abs(
            predicted_df["Moisture1_Raw"]
            -
            predicted_df["Moisture2_Raw"]
        )

        predicted_df["Moisture_Spread"] = (
            predicted_df["Moisture_Diff"]
        )

        max_diff = (
            predicted_df["Moisture_Diff"].max()
        )

        if (
            pd.notna(max_diff)
            and
            max_diff != 0
        ):

            predicted_df["Uniformity"] = (
                1
                -
                (
                    predicted_df["Moisture_Diff"]
                    / max_diff
                )
            )

    predicted_df[
        "Simulation_Water_Added_ml"
    ] = water_increment

    predicted_df[
        "Simulation_Total_Water_ml"
    ] = total_water

    predicted_df[
        "Simulation_Soil_Type"
    ] = soil_type

    predicted_df[
        "Prediction_Timestamp"
    ] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return predicted_df


# ============================================================
# LOAD LIVE DATA & SIDEBAR SETUP
# ============================================================

live_df = load_csv(LIVE_CSV)

st.sidebar.title("🌱 Soil Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📡 Real-Time Monitoring",
        "🔮 Virtual Spectrometer (Live Twin)",
        "📡 THz Spectroscopy Analysis",
        "📁 Upload Dataset",
        "📈 Experimental Graphs",
        "📊 Calibration Studio",
        "🔮 Digital Twin Simulator",
        "🧪 Save Experiment",
        "📂 Project Storage",
        "🗂️ Manage Datasets"
    ]
)

st.sidebar.markdown("---")

st.sidebar.header("⚙️ Experiment Settings")

SOIL_TYPE = st.sidebar.selectbox(
    "Soil Type",
    [
        "Natural Loamy",
        "Dried Loamy"
    ]
)

WATER_ADDED = st.sidebar.number_input(
    "Current Water Added (mL)",
    min_value=0.0,
    value=0.0,
    step=2.5
)

EXPERIMENT_ID = st.sidebar.text_input(
    "Experiment ID",
    "EXP001"
)

# ============================================================
# 1. HOME DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🌱 Soil Intelligence Digital Twin</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Real-time telemetry, dual soil moisture sensing, "
        "dielectric calibration, and digital twin simulation."
    )

    st.markdown("---")

    if live_df.empty:
        st.warning(
            "No live sensor data found in csv/live_data.csv. "
            "Ensure the ESP32 is online and wifi_reader.py is running."
        )

    else:
        row = live_df.iloc[-1]

        def sensor_value(column, default=0.0):
            value = pd.to_numeric(
                row.get(column, default),
                errors="coerce"
            )
            return float(value) if pd.notna(value) else default

        moisture1 = sensor_value("Moisture1_Raw")
        moisture2 = sensor_value("Moisture2_Raw")

        moisture1_v = sensor_value("Moisture1_V")
        moisture2_v = sensor_value("Moisture2_V")

        moisture_avg = sensor_value(
            "Moisture_Avg",
            (moisture1 + moisture2) / 2
        )

        moisture_diff = sensor_value(
            "Moisture_Diff",
            abs(moisture1 - moisture2)
        )

        moisture_spread = sensor_value(
            "Moisture_Spread",
            moisture_diff
        )

        uniformity = sensor_value("Uniformity")

        temp_c = sensor_value("Temperature_C")
        ph_v = sensor_value("pH_Voltage")
        ec = sensor_value("EC_uS_cm")

        st.subheader("📡 Latest Sensor Readings")

        c1, c2, c3 = st.columns(3)

        c1.metric("💧 Moisture Sensor 1", f"{moisture1:.0f} ADC")
        c2.metric("💧 Moisture Sensor 2", f"{moisture2:.0f} ADC")
        c3.metric("💧 Average Moisture", f"{moisture_avg:.1f} ADC")

        c4, c5, c6 = st.columns(3)

        c4.metric("📏 Moisture Difference", f"{moisture_diff:.1f} ADC")
        c5.metric("📊 Uniformity", f"{uniformity:.3f}")
        c6.metric("🌡 Temperature", f"{temp_c:.2f} °C")

        c7, c8, c9 = st.columns(3)

        c7.metric("Sensor 1 Voltage", f"{moisture1_v:.3f} V")
        c8.metric("Sensor 2 Voltage", f"{moisture2_v:.3f} V")
        c9.metric("🧪 pH Voltage", f"{ph_v:.3f} V")

        st.metric("⚡ EC Conductivity", f"{ec:.1f} µS/cm")

        st.markdown("---")

        moisture_columns = [
            col for col in [
                "Moisture1_Raw",
                "Moisture2_Raw",
                "Moisture_Avg"
            ]
            if col in live_df.columns
        ]

        if moisture_columns:
            st.subheader("💧 Dual Moisture Sensor Comparison")

            st.line_chart(
                live_df[moisture_columns].apply(
                    pd.to_numeric, errors="coerce"
                ),
                use_container_width=True
            )

        trend_columns = [
            col for col in [
                "Temperature_C",
                "pH_Voltage",
                "EC_uS_cm"
            ]
            if col in live_df.columns
        ]

        if trend_columns:
            st.subheader("📈 Other Sensor Trends")

            st.line_chart(
                live_df[trend_columns].apply(
                    pd.to_numeric, errors="coerce"
                ),
                use_container_width=True
            )

        st.subheader("📄 Latest Telemetry Record")
        st.dataframe(
            live_df.tail(10),
            use_container_width=True
        )

        st.download_button(
            "📥 Download Live Session CSV",
            live_df.to_csv(index=False).encode("utf-8"),
            file_name="live_soil_data.csv",
            mime="text/csv",
            key="dashboard_download_live"
        )


# ============================================================
# 2. REAL-TIME MONITORING
# ============================================================

elif page == "📡 Real-Time Monitoring":

    st.markdown(
        '<div class="section-title">📡 Real-Time Soil Monitoring</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        auto_refresh = st.checkbox(
            "🔄 Auto Refresh (Every 2 seconds)",
            value=True,
            key="realtime_refresh"
        )

    with col2:
        if st.button(
            "🚨 End Process & Start Fresh Session",
            key="restart_live_session"
        ):
            reset_live_session()
            st.success("Live CSV session reset.")
            st.rerun()

    if live_df.empty:
        st.warning(
            "No real-time data found. Please run wifi_reader.py "
            "and verify that the ESP32 is sending telemetry."
        )

    else:
        row = live_df.iloc[-1]

        def latest_sensor_value(column, default=0.0):
            value = pd.to_numeric(
                row.get(column, default),
                errors="coerce"
            )
            return float(value) if pd.notna(value) else default

        m1 = latest_sensor_value("Moisture1_Raw")
        m2 = latest_sensor_value("Moisture2_Raw")
        v1 = latest_sensor_value("Moisture1_V")
        v2 = latest_sensor_value("Moisture2_V")

        avg = latest_sensor_value(
            "Moisture_Avg", (m1 + m2) / 2
        )

        diff = latest_sensor_value(
            "Moisture_Diff", abs(m1 - m2)
        )

        uniformity = latest_sensor_value("Uniformity")
        temp = latest_sensor_value("Temperature_C")
        ph = latest_sensor_value("pH_Voltage")
        ec = latest_sensor_value("EC_uS_cm")

        st.success("📡 Live telemetry data loaded.")

        st.subheader("💧 Soil Moisture Sensors")

        c1, c2, c3 = st.columns(3)

        c1.metric("Moisture Sensor 1", f"{m1:.0f} ADC")
        c2.metric("Moisture Sensor 2", f"{m2:.0f} ADC")
        c3.metric("Average Moisture", f"{avg:.1f} ADC")

        c4, c5, c6 = st.columns(3)

        c4.metric("Sensor 1 Voltage", f"{v1:.3f} V")
        c5.metric("Sensor 2 Voltage", f"{v2:.3f} V")
        c6.metric("Moisture Difference", f"{diff:.1f} ADC")

        c7, c8, c9 = st.columns(3)

        c7.metric("Uniformity", f"{uniformity:.3f}")
        c8.metric("Temperature", f"{temp:.2f} °C")
        c9.metric("pH Voltage", f"{ph:.3f} V")

        st.metric("EC Conductivity", f"{ec:.1f} µS/cm")

        st.markdown("---")

        st.subheader("📊 Live Data Table (Last 30 Records)")

        st.dataframe(
            live_df.tail(30),
            use_container_width=True
        )

        moisture_columns = [
            col for col in [
                "Moisture1_Raw",
                "Moisture2_Raw",
                "Moisture_Avg"
            ]
            if col in live_df.columns
        ]

        if moisture_columns:
            st.subheader("📈 Moisture Sensor Comparison")

            st.line_chart(
                live_df[moisture_columns].apply(
                    pd.to_numeric, errors="coerce"
                ),
                use_container_width=True
            )

        numeric_columns = get_numeric_columns(live_df)

        selected_columns = st.multiselect(
            "Select Parameters to Graph",
            numeric_columns,
            default=[
                col for col in [
                    "Moisture1_Raw",
                    "Moisture2_Raw",
                    "Moisture_Avg"
                ]
                if col in numeric_columns
            ],
            key="live_parameters"
        )

        if selected_columns:
            fig = px.line(
                live_df,
                y=selected_columns,
                title="Live Sensor Traces Over Time"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.download_button(
            "📥 Download Current Real-Time CSV",
            live_df.to_csv(index=False).encode("utf-8"),
            file_name="realtime_soil_session.csv",
            mime="text/csv",
            key="download_live"
        )

    if auto_refresh:
        time.sleep(2)
        st.rerun()

# ============================================================
# 3. UPLOAD DATASET
# ============================================================

elif page == "📁 Upload Dataset":

    st.markdown(
        '<div class="section-title">📁 Upload Dataset</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Laboratory or Experimental CSV File",
        type=["csv"],
        key="dataset_upload"
    )

    if uploaded_file:

        try:

            upload_df = pd.read_csv(
                uploaded_file
            )

            file_path = os.path.join(
                UPLOAD_DIR,
                uploaded_file.name
            )

            upload_df.to_csv(
                file_path,
                index=False
            )

            st.success(
                f"✅ {uploaded_file.name} saved successfully in `{UPLOAD_DIR}`!"
            )

            st.dataframe(
                upload_df.head(20),
                use_container_width=True
            )

            st.download_button(
                f"📥 Download {uploaded_file.name}",
                upload_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=uploaded_file.name,
                mime="text/csv",
                key="download_just_uploaded"
            )

        except Exception as e:

            st.error(
                f"❌ Error uploading file: {e}"
            )

    st.markdown("---")

    st.subheader(
        "📂 Stored Uploaded Datasets"
    )

    uploaded_files = [
        f
        for f in os.listdir(UPLOAD_DIR)
        if f.lower().endswith(".csv")
    ]

    if uploaded_files:

        for f_name in uploaded_files:

            col_f, col_d = st.columns([3, 1])

            col_f.write(
                f"📄 **{f_name}**"
            )

            d_df = load_csv(
                os.path.join(
                    UPLOAD_DIR,
                    f_name
                )
            )

            col_d.download_button(
                "📥 Download",
                d_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f_name,
                key=f"dl_{f_name}"
            )

    else:

        st.info(
            "No uploaded datasets present."
        )

# ============================================================
# 4. EXPERIMENTAL GRAPHS
# ============================================================

elif page == "📈 Experimental Graphs":

    st.markdown(
        '<div class="section-title">📈 Experimental Graphs</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload ZIP files containing your generated experimental graphs. "
        "The dashboard will automatically extract and visualize the graphs."
    )

    # ========================================================
    # NEW: ZIP UPLOAD
    # ========================================================

    st.subheader(
        "📦 Upload Graph ZIP Files"
    )

    uploaded_graph_zips = st.file_uploader(
        "Upload one or more ZIP files containing your graphs",
        type=["zip"],
        accept_multiple_files=True,
        key="experimental_graph_zip_upload"
    )

    if uploaded_graph_zips:

        for uploaded_zip in uploaded_graph_zips:

            zip_name = os.path.splitext(
                uploaded_zip.name
            )[0]

            # Keep each uploaded ZIP separate
            destination = os.path.join(
                UPLOADED_GRAPH_DIR,
                zip_name
            )

            marker_file = os.path.join(
                destination,
                ".uploaded"
            )

            if not os.path.exists(marker_file):

                try:

                    # If the same collection exists without
                    # a marker, remove it first.
                    if os.path.exists(destination):
                        shutil.rmtree(destination)

                    os.makedirs(
                        destination,
                        exist_ok=True
                    )

                    safe_extract_zip(
                        uploaded_zip,
                        destination
                    )

                    with open(
                        marker_file,
                        "w",
                        encoding="utf-8"
                    ) as marker:

                        marker.write(
                            uploaded_zip.name
                        )

                    st.success(
                        f"✅ {uploaded_zip.name} uploaded and extracted successfully."
                    )

                except zipfile.BadZipFile:

                    st.error(
                        f"❌ {uploaded_zip.name} is not a valid ZIP file."
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error extracting {uploaded_zip.name}: {e}"
                    )

    st.markdown("---")

    # ========================================================
    # UPLOADED GRAPH COLLECTIONS
    # ========================================================

    uploaded_collections = (
        get_uploaded_graph_collections()
    )

    if uploaded_collections:

        st.subheader(
            "📂 Uploaded Graph Collections"
        )

        selected_collection = st.selectbox(
            "Select Uploaded ZIP",
            uploaded_collections,
            key="uploaded_graph_collection"
        )

        # Get all graphs inside selected ZIP
        graph_files = get_uploaded_graph_files(
            selected_collection
        )

        if not graph_files:

            st.warning(
                "⚠️ No PNG, JPG, JPEG, or WEBP graph files "
                "were found inside this ZIP."
            )

        else:

            st.success(
                f"✅ {len(graph_files)} graph(s) found in "
                f"`{selected_collection}.zip`"
            )

            # =================================================
            # CATEGORY FILTER
            # =================================================

            folders = sorted(
                set(
                    graph["folder"]
                    for graph in graph_files
                )
            )

            folder_options = [
                "All"
            ] + folders

            selected_folder = st.selectbox(
                "📁 Select Graph Folder / Category",
                folder_options,
                key="uploaded_graph_folder"
            )

            if selected_folder == "All":

                filtered_graphs = graph_files

            else:

                filtered_graphs = [
                    graph
                    for graph in graph_files
                    if graph["folder"] == selected_folder
                ]

            # =================================================
            # SEARCH GRAPH
            # =================================================

            search_text = st.text_input(
                "🔎 Search Graph Name",
                "",
                key="uploaded_graph_search"
            )

            if search_text.strip():

                search_lower = (
                    search_text.strip().lower()
                )

                filtered_graphs = [
                    graph
                    for graph in filtered_graphs
                    if search_lower in graph["name"].lower()
                    or search_lower in graph["relative"].lower()
                ]

            st.write(
                f"Showing **{len(filtered_graphs)}** graph(s)"
            )

            # =================================================
            # INDIVIDUAL GRAPH VIEWER
            # =================================================

            if filtered_graphs:

                graph_options = [
                    graph["relative"]
                    for graph in filtered_graphs
                ]

                selected_graph_name = st.selectbox(
                    "📊 Select Graph to Visualize",
                    graph_options,
                    key="uploaded_graph_selected"
                )

                selected_graph = next(
                    graph
                    for graph in filtered_graphs
                    if graph["relative"]
                    == selected_graph_name
                )

                st.markdown("---")

                st.subheader(
                    f"📈 {selected_graph['name']}"
                )

                st.caption(
                    f"Path inside ZIP: "
                    f"{selected_graph['relative']}"
                )

                st.image(
                    selected_graph["path"],
                    use_container_width=True
                )

                # =================================================
                # DOWNLOAD SELECTED GRAPH
                # =================================================

                with open(
                    selected_graph["path"],
                    "rb"
                ) as graph_file:

                    graph_bytes = graph_file.read()

                extension = os.path.splitext(
                    selected_graph["name"]
                )[1].lower()

                mime_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp"
                }.get(
                    extension,
                    "application/octet-stream"
                )

                st.download_button(
                    "⬇️ Download This Graph",
                    data=graph_bytes,
                    file_name=selected_graph["name"],
                    mime=mime_type,
                    key=(
                        "download_uploaded_graph_"
                        + selected_collection
                        + "_"
                        + selected_graph_name
                    )
                )

            else:

                st.warning(
                    "No graphs match the selected search/filter."
                )

            # =================================================
            # DISPLAY ALL FILTERED GRAPHS
            # =================================================

            st.markdown("---")

            show_all_graphs = st.checkbox(
                "🖼️ Display All Filtered Graphs",
                value=False,
                key="display_all_uploaded_graphs"
            )

            if show_all_graphs:

                st.subheader(
                    "🖼️ All Filtered Graphs"
                )

                for index, graph in enumerate(
                    filtered_graphs
                ):

                    st.markdown(
                        f"### {index + 1}. {graph['name']}"
                    )

                    st.caption(
                        graph["relative"]
                    )

                    st.image(
                        graph["path"],
                        use_container_width=True
                    )

                    with open(
                        graph["path"],
                        "rb"
                    ) as graph_file:

                        graph_bytes = graph_file.read()

                    extension = os.path.splitext(
                        graph["name"]
                    )[1].lower()

                    mime_type = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp"
                    }.get(
                        extension,
                        "application/octet-stream"
                    )

                    st.download_button(
                        "⬇️ Download Graph",
                        data=graph_bytes,
                        file_name=graph["name"],
                        mime=mime_type,
                        key=(
                            "download_all_uploaded_graph_"
                            + selected_collection
                            + "_"
                            + str(index)
                        )
                    )

                    st.divider()

    else:

        st.info(
            "📂 No graph ZIP files have been uploaded yet. "
            "Upload a ZIP above to visualize its graphs."
        )

    # ========================================================
    # EXISTING GRAPHS ALREADY IN GRAPH DIRECTORY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Existing Graphs in Dashboard"
    )

    graph_groups = get_graph_groups()

    # Do not show the uploaded container as an existing
    # graph collection here.
    existing_graph_groups = {
        name: categories
        for name, categories in graph_groups.items()
        if name != "uploaded"
    }

    if not existing_graph_groups:

        st.info(
            "No pre-existing graph folders were found."
        )

    else:

        selected_existing_group = st.selectbox(
            "Select Existing Experiment Graph Set",
            list(existing_graph_groups.keys()),
            key="experimental_existing_graph_group"
        )

        display_graph_gallery(
            selected_existing_group,
            existing_graph_groups[
                selected_existing_group
            ]
        )

# ============================================================
# 5. CALIBRATION STUDIO
# ============================================================

elif page == "📊 Calibration Studio":

    st.markdown(
        '<div class="section-title">📊 Calibration Studio</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Extracts physical calibration curves and dielectric responses directly from selected datasets."
    )

    datasets = get_available_datasets(
        live_df
    )

    if not datasets:

        st.warning(
            "⚠️ No datasets available. Upload files or collect real-time data first."
        )

    else:

        dataset_names = list(
            datasets.keys()
        )

        selected_datasets = st.multiselect(
            "Select Datasets for Calibration Comparison",
            dataset_names,
            default=[
                dataset_names[0]
            ]
            if dataset_names
            else []
        )

        if selected_datasets:

            st.markdown(
                "##### 📥 Download Selected Datasets Individually:"
            )

            d_cols = st.columns(
                min(len(selected_datasets), 4)
            )

            for idx, d_name in enumerate(
                selected_datasets
            ):

                df_to_download = datasets[
                    d_name
                ]["df"]

                clean_name = datasets[
                    d_name
                ]["filename"]

                with d_cols[
                    idx % 4
                ]:

                    st.download_button(
                        label=f"📥 Download {clean_name}",
                        data=df_to_download.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name=clean_name,
                        mime="text/csv",
                        key=f"calib_dl_{d_name}"
                    )

            fig_moist = go.Figure()
            fig_moist_pct = go.Figure()
            fig_ec = go.Figure()
            fig_temp = go.Figure()
            fig_ph = go.Figure()

            colors = [
                '#d32f2f',
                '#1976d2',
                '#388e3c',
                '#f57c00',
                '#7b1fa2',
                '#0097a7'
            ]

            stats = {
                'moist': {},
                'ec': {},
                'temp': {},
                'ph': {}
            }

            for i, d_name in enumerate(
                selected_datasets
            ):

                df = datasets[
                    d_name
                ]["df"].copy()

                c = colors[
                    i % len(colors)
                ]

                if "Water_Added_ml" in df.columns:

                    df = df.sort_values(
                        'Water_Added_ml'
                    )

                    x_col = "Water_Added_ml"

                    if "Moisture_Avg" in df.columns:

                        df[
                            "Moisture_Avg"
                        ] = pd.to_numeric(
                            df["Moisture_Avg"],
                            errors='coerce'
                        )

                        stats[
                            'moist'
                        ][d_name] = {
                            'min':
                                df["Moisture_Avg"].min(),
                            'max':
                                df["Moisture_Avg"].max(),
                            'drop':
                                (
                                    df["Moisture_Avg"].iloc[0]
                                    -
                                    df["Moisture_Avg"].iloc[-1]
                                )
                                if len(df) > 0
                                else 0
                        }

                        fig_moist.add_trace(
                            go.Scatter(
                                x=df[x_col],
                                y=df["Moisture_Avg"],
                                mode='lines+markers',
                                name=f"{d_name} (ADC)",
                                line=dict(
                                    color=c,
                                    width=3
                                ),
                                marker=dict(
                                    size=10
                                )
                            )
                        )

                        df[
                            "Moisture_Pct"
                        ] = (
                            (
                                4095
                                -
                                df["Moisture_Avg"]
                            )
                            /
                            (4095 - 817)
                        ) * 96.9

                        fig_moist_pct.add_trace(
                            go.Scatter(
                                x=df[x_col],
                                y=df["Moisture_Pct"],
                                mode='lines+markers',
                                name=f"{d_name} (%)",
                                line=dict(
                                    color=c,
                                    width=3
                                ),
                                marker=dict(
                                    size=10,
                                    symbol='square'
                                )
                            )
                        )

                    ec_col = (
                        "EC_uS_cm"
                        if "EC_uS_cm" in df.columns
                        else (
                            "TDS_Value"
                            if "TDS_Value" in df.columns
                            else None
                        )
                    )

                    if ec_col:

                        df[ec_col] = pd.to_numeric(
                            df[ec_col],
                            errors='coerce'
                        )

                        stats[
                            'ec'
                        ][d_name] = {
                            'max':
                                df[ec_col].max(),
                            'min':
                                df[ec_col].min()
                        }

                        fig_ec.add_trace(
                            go.Scatter(
                                x=df[x_col],
                                y=df[ec_col],
                                mode='lines+markers',
                                name=f"{d_name} (EC)",
                                line=dict(
                                    color=c,
                                    width=3
                                ),
                                marker=dict(
                                    size=10,
                                    symbol='triangle-up'
                                )
                            )
                        )

                    if "Temperature_C" in df.columns:

                        df[
                            "Temperature_C"
                        ] = pd.to_numeric(
                            df["Temperature_C"],
                            errors='coerce'
                        )

                        if len(df) > 0:

                            stats[
                                'temp'
                            ][d_name] = {
                                'start':
                                    df["Temperature_C"].iloc[0],
                                'end':
                                    df["Temperature_C"].iloc[-1]
                            }

                        fig_temp.add_trace(
                            go.Scatter(
                                x=df[x_col],
                                y=df["Temperature_C"],
                                mode='lines+markers',
                                name=f"{d_name} (°C)",
                                line=dict(
                                    color=c,
                                    width=3
                                ),
                                marker=dict(
                                    size=10
                                )
                            )
                        )

                    if "pH_Voltage" in df.columns:

                        df[
                            "pH_Voltage"
                        ] = pd.to_numeric(
                            df["pH_Voltage"],
                            errors='coerce'
                        )

                        stats[
                            'ph'
                        ][d_name] = {
                            'var':
                                df["pH_Voltage"].max()
                                -
                                df["pH_Voltage"].min()
                        }

                        fig_ph.add_trace(
                            go.Scatter(
                                x=df[x_col],
                                y=df["pH_Voltage"],
                                mode='lines+markers',
                                name=f"{d_name} (V)",
                                line=dict(
                                    color=c,
                                    width=3
                                ),
                                marker=dict(
                                    size=10,
                                    symbol='diamond'
                                )
                            )
                        )

            layout_args = dict(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#eeeeee',
                    zeroline=False,
                    title="Water Injected into Soil (mL)",
                    title_font=dict(
                        size=14,
                        color='black'
                    )
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#eeeeee',
                    zeroline=False,
                    title_font=dict(
                        size=14,
                        color='black'
                    )
                ),
                margin=dict(
                    l=60,
                    r=40,
                    t=60,
                    b=60
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            fig_moist.update_layout(
                title="<b>1. Soil Moisture Calibration: Raw ADC vs Water Injected</b>",
                yaxis_title="Raw ADC Reading",
                **layout_args
            )

            st.plotly_chart(
                fig_moist,
                use_container_width=True
            )

            fig_moist_pct.update_layout(
                title="<b>2. Calibrated Volumetric Soil Moisture (%) vs Water Injected</b>",
                yaxis_title="Soil Moisture (%)",
                **layout_args
            )

            st.plotly_chart(
                fig_moist_pct,
                use_container_width=True
            )

            fig_ec.update_layout(
                title="<b>3. Bulk Electrical Conductivity (EC) vs Water Injected</b>",
                yaxis_title="Electrical Conductivity (µS/cm)",
                **layout_args
            )

            st.plotly_chart(
                fig_ec,
                use_container_width=True
            )

            fig_temp.update_layout(
                title="<b>4. Soil Temperature vs Water Injected (Evaporative Cooling)</b>",
                yaxis_title="Soil Temperature (°C)",
                **layout_args
            )

            st.plotly_chart(
                fig_temp,
                use_container_width=True
            )

            fig_ph.update_layout(
                title="<b>5. Electrochemical Stability (pH) vs Water Injected</b>",
                yaxis_title="pH Sensor Voltage (V)",
                **layout_args
            )

            st.plotly_chart(
                fig_ph,
                use_container_width=True
            )

# ============================================================
# 6. DIGITAL TWIN SIMULATOR
# ============================================================

elif page == "🔮 Digital Twin Simulator":

    st.markdown(
        '<div class="section-title">🔮 Digital Twin Simulator</div>',
        unsafe_allow_html=True
    )

    datasets = get_available_datasets(
        live_df
    )

    if not datasets:

        st.warning(
            "⚠️ No dataset available for prediction."
        )

    else:

        base_name = st.selectbox(
            "Select Baseline Dataset for Simulation",
            list(datasets.keys()),
            key="prediction_base"
        )

        base_df = datasets[
            base_name
        ]["df"].copy()

        st.download_button(
            label=f"📥 Download Baseline Dataset ({datasets[base_name]['filename']})",
            data=base_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=datasets[
                base_name
            ]["filename"],
            mime="text/csv",
            key="download_base_dataset"
        )

        moisture_col = get_moisture_column(
            base_df
        )

        current_moisture = get_latest_value(
            base_df,
            moisture_col
        )

        current_temp = get_latest_value(
            base_df,
            "Temperature_C"
        )

        current_ph = get_latest_value(
            base_df,
            "pH_Voltage"
        )

        current_uniformity = get_latest_value(
            base_df,
            "Uniformity"
        )

        st.subheader(
            "📡 Current Baseline Soil Values"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "💧 Moisture",
            f"{current_moisture:.0f} ADC"
        )

        c2.metric(
            "🌡 Temperature",
            f"{current_temp:.2f} °C"
        )

        c3.metric(
            "🧪 pH Voltage",
            f"{current_ph:.3f} V"
        )

        c4.metric(
            "📊 Uniformity",
            f"{current_uniformity:.3f}"
        )

        water_increment = st.number_input(
            "💧 Simulated Water Increment (mL)",
            min_value=0.0,
            value=25.0,
            step=5.0,
            key="water_increment"
        )

        total_water = (
            WATER_ADDED
            +
            water_increment
        )

        if st.button(
            "🔮 Run Digital Twin Simulation",
            type="primary",
            use_container_width=True,
            key="predict_experiment"
        ):

            predicted_df = generate_predicted_dataset(
                base_df,
                water_increment,
                SOIL_TYPE,
                total_water
            )

            if not predicted_df.empty:

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"{EXPERIMENT_ID}"
                    f"_PREDICTED_"
                    f"{int(total_water)}ml_"
                    f"{timestamp}.csv"
                )

                path = os.path.join(
                    PREDICTION_DIR,
                    filename
                )

                predicted_df.to_csv(
                    path,
                    index=False
                )

                st.session_state.last_prediction = (
                    predicted_df
                )

                st.session_state.last_prediction_name = (
                    filename
                )

                st.success(
                    "✅ Prediction generated and saved!"
                )

        if not st.session_state.last_prediction.empty:

            predicted_df = (
                st.session_state.last_prediction
            )

            predicted_moisture = get_latest_value(
                predicted_df,
                get_moisture_column(
                    predicted_df
                )
            )

            predicted_temp = get_latest_value(
                predicted_df,
                "Temperature_C"
            )

            predicted_ph = get_latest_value(
                predicted_df,
                "pH_Voltage"
            )

            predicted_uniformity = get_latest_value(
                predicted_df,
                "Uniformity"
            )

            st.markdown("---")

            st.subheader(
                "🔮 Simulated State Transition Results"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "💧 Predicted Moisture",
                f"{predicted_moisture:.0f} ADC",
                delta=f"{predicted_moisture - current_moisture:.0f}"
            )

            c2.metric(
                "🌡 Predicted Temperature",
                f"{predicted_temp:.2f} °C",
                delta=f"{predicted_temp - current_temp:.2f}"
            )

            c3.metric(
                "🧪 Predicted pH",
                f"{predicted_ph:.3f} V",
                delta=f"{predicted_ph - current_ph:.3f}"
            )

            c4.metric(
                "📊 Predicted Uniformity",
                f"{predicted_uniformity:.3f}",
                delta=f"{predicted_uniformity - current_uniformity:.3f}"
            )

            st.subheader(
                "📄 Simulated State Data Table"
            )

            st.dataframe(
                predicted_df.tail(20),
                use_container_width=True
            )

            st.download_button(
                "📥 Download This Predicted CSV",
                predicted_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=st.session_state.last_prediction_name,
                mime="text/csv",
                key="download_prediction"
            )

# ============================================================
# 7. SAVE EXPERIMENT
# ============================================================


# ============================================================
# VIRTUAL SPECTROMETER & CROSS-MODAL DIGITAL TWIN
# ============================================================

elif page == "🔮 Virtual Spectrometer (Live Twin)":

    st.markdown(
        '<div class="section-title">🔮 Virtual Spectrometer & Cross-Modal Digital Twin</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='color:#555555; font-size:15px; margin-top:-10px; margin-bottom:20px;'>
        <b>Multimodal Cyber-Physical Observer:</b> Infers soil matrix category, grain size, volumetric hydration, 
        and microplastics (LDPE) contamination percentage directly from live in-situ IoT telemetry, 
        and reconstructs laboratory <b>Terahertz Time-Domain Spectroscopy (THz-TDS)</b> optical parameters in real time.
        </p>
        """,
        unsafe_allow_html=True
    )

    cm_bundle = load_cross_modal_models()
    if cm_bundle is None:
        st.error("⚠️ Cross-modal models not found. Please ensure models exist in `dashboard/models/`.")
    else:
        clf = cm_bundle["clf"]
        le = cm_bundle["le"]
        reg_w = cm_bundle["reg_w"]
        reg_l = cm_bundle["reg_l"]
        reg_thz = cm_bundle["reg_thz"]
        meta = cm_bundle["meta"]

        # Telemetry Source Selection (Interactive Preset vs Wi-Fi SoftAP vs USB Serial vs Log File)
        src_col1, src_col2 = st.columns([1, 2])
        with src_col1:
            input_mode = st.radio(
                "📡 Telemetry Data Source",
                [
                    "🧪 Interactive Preset & Studio",
                    "📶 Live ESP32-S3 Wi-Fi SoftAP (http://192.168.4.1)",
                    "🔌 Live ESP32-S3 Hardware Feed (USB-C Serial)",
                    "📁 Live Ingestion Log (live_data.csv)"
                ],
                key="vm_input_mode"
            )

        # Default values in session state
        if "esp_moist_v" not in st.session_state:
            st.session_state["esp_moist_v"] = 3.30
            st.session_state["esp_ec"] = 0.0
            st.session_state["esp_ph_v"] = 1.855
            st.session_state["esp_temp"] = 28.0
            st.session_state["esp_infer_us"] = 23

        v_moist = st.session_state["esp_moist_v"]
        ec_val = st.session_state["esp_ec"]
        v_ph = st.session_state["esp_ph_v"]
        temp_val = st.session_state["esp_temp"]

        preset_data = {
            "Oven-Dried Soil / Air Reference Baseline (0 mL - 3.30V)": [3.30, 0.0, 2.848, 30.19],
            "Natural Loamy (Dry - 0 mL)": [3.30, 0.0, 1.855, 28.0],
            "Natural Loamy (Hydrated - 7.5 mL)": [0.877, 519.1, 1.785, 27.5],
            "Oven-Dried Pure Soil (<300 µm, Dry)": [3.30, 0.0, 2.642, 29.5],
            "Oven-Dried Pure Soil (<600 µm, 10 mL)": [0.887, 336.4, 2.162, 28.5],
            "LDPE Contaminated 2.0% (<300 µm Soil)": [0.819, 669.8, 2.550, 29.0],
            "LDPE Contaminated 4.0% (<600 µm Soil)": [0.849, 617.2, 2.561, 29.0]
        }

        if input_mode == "🧪 Interactive Preset & Studio":
            with src_col2:
                selected_preset = st.selectbox(
                    "⚡ Quick Scenario Presets (Pre-calibrated Experimental States)",
                    list(preset_data.keys()),
                    key="vm_preset_select"
                )
                vec = preset_data[selected_preset]
                v_moist, ec_val, v_ph, temp_val = vec[0], vec[1], vec[2], vec[3]

            st.markdown("##### 🎛️ Fine-Tune In-Situ Sensor Telemetry:")
            sl1, sl2, sl3, sl4 = st.columns(4)
            with sl1:
                v_moist = st.slider("💧 Capacitive Voltage (V)", 0.50, 3.30, float(v_moist), 0.005)
            with sl2:
                ec_val = st.slider("⚡ EC (µS/cm)", 0.0, 2500.0, float(ec_val), 1.0)
            with sl3:
                v_ph = st.slider("🧪 pH Sensor Voltage (V)", 1.50, 2.85, float(v_ph), 0.005)
            with sl4:
                temp_val = st.slider("🌡 Temperature (°C)", 20.0, 38.0, float(temp_val), 0.1)

        elif input_mode == "📶 Live ESP32-S3 Wi-Fi SoftAP (http://192.168.4.1)":
            with src_col2:
                st.markdown("##### 📶 ESP32-S3 Autonomous SoftAP Wi-Fi Link (Offline / No Internet):")
                w_c1, w_c2, w_c3 = st.columns([2, 1, 1])
                with w_c1:
                    wifi_url = st.text_input("SoftAP REST Endpoint", "http://192.168.4.1/api/data", key="vm_wifi_url")
                with w_c2:
                    st.write("")
                    st.write("")
                    wifi_fetch_btn = st.button("⚡ Fetch Wi-Fi", type="primary", key="vm_wifi_fetch")
                with w_c3:
                    st.write("")
                    st.write("")
                    wifi_demo_btn = st.button("🧪 Demo Wi-Fi", key="vm_wifi_demo")

                if wifi_fetch_btn:
                    try:
                        import requests
                        res = requests.get(wifi_url, timeout=2.5)
                        if res.status_code == 200:
                            d = res.json()
                            st.session_state["esp_moist_v"] = float(d.get("m1_v", 2.5))
                            st.session_state["esp_ec"] = float(d.get("ec_comp", 100.0))
                            st.session_state["esp_ph_v"] = float(d.get("ph_v", 1.85))
                            st.session_state["esp_temp"] = float(d.get("temp_c", 27.5))
                            st.session_state["esp_infer_us"] = int(d.get("infer_us", 23))
                            st.success(f"✅ Live Wi-Fi Telemetry Acquired from {wifi_url}! On-Chip Edge AI Latency: {st.session_state['esp_infer_us']} µs")
                        else:
                            st.error(f"HTTP Error {res.status_code} from {wifi_url}")
                    except Exception as e:
                        st.error(f"Wi-Fi Connection Failed: {e}. Connect laptop to Wi-Fi 'SOIL_DIAGNOSTIC_NODE'.")

                if wifi_demo_btn:
                    import random
                    sim_scenarios = [
                        [0.656, 2164.8, 1.750, 27.47, "Natural Loamy Saturated (10 mL)"],
                        [1.350, 834.7, 1.810, 27.91, "Natural Loamy Medium (5 mL)"],
                        [3.300, 0.0, 1.860, 28.62, "Natural Loamy Dry (0 mL)"],
                        [0.887, 54.3, 2.228, 28.71, "LDPE Microplastic Contaminated 4.0%"]
                    ]
                    pick = random.choice(sim_scenarios)
                    st.session_state["esp_moist_v"] = pick[0]
                    st.session_state["esp_ec"] = pick[1]
                    st.session_state["esp_ph_v"] = pick[2]
                    st.session_state["esp_temp"] = pick[3]
                    st.session_state["esp_infer_us"] = random.randint(21, 26)
                    st.success(f"✅ Simulated Wi-Fi Packet: {pick[4]} | On-Chip Latency: {st.session_state['esp_infer_us']} µs")

                v_moist = st.session_state["esp_moist_v"]
                ec_val = st.session_state["esp_ec"]
                v_ph = st.session_state["esp_ph_v"]
                temp_val = st.session_state["esp_temp"]

                st.info(
                    f"📶 **SoftAP Wi-Fi Link Active** | "
                    f"**Moisture Pin (GPIO 5):** `{v_moist:.3f} V` | **EC Pin (GPIO 7):** `{ec_val:.1f} µS/cm` | "
                    f"**pH Pin (GPIO 6):** `{v_ph:.3f} V` | **Temp (GPIO 4):** `{temp_val:.2f} °C` | "
                    f"**On-Chip Latency:** `{st.session_state['esp_infer_us']} µs`"
                )

        elif input_mode == "🔌 Live ESP32-S3 Hardware Feed (USB-C Serial)":
            with src_col2:
                st.markdown("##### 🔌 ESP32-S3 Edge AI Hardware Link:")
                try:
                    import serial
                    import serial.tools.list_ports
                    detected_ports = [p.device for p in serial.tools.list_ports.comports() if "Bluetooth" not in p.description]
                except Exception:
                    detected_ports = []

                port_options = detected_ports + ["🧪 Simulated ESP32-S3 Virtual Stream (Demo)"]
                
                h_c1, h_c2, h_c3 = st.columns([2, 1, 1])
                with h_c1:
                    chosen_port = st.selectbox("Select Hardware Port", port_options, index=0)
                with h_c2:
                    baud = st.selectbox("Baudrate", [115200, 9600, 57600], index=0)
                with h_c3:
                    st.write("")
                    st.write("")
                    poll_btn = st.button("⚡ Read ESP32-S3", type="primary")

                if poll_btn:
                    if chosen_port == "🧪 Simulated ESP32-S3 Virtual Stream (Demo)":
                        import random
                        sim_scenarios = [
                            [0.656, 2164.8, 1.750, 27.47, "Natural Loamy Saturated"],
                            [1.350, 834.7, 1.810, 27.91, "Natural Loamy Medium (5 mL)"],
                            [3.300, 0.0, 1.860, 28.62, "Natural Clean Dry"],
                            [0.887, 54.3, 2.228, 28.71, "LDPE Contaminated 4.0%"]
                        ]
                        pick = random.choice(sim_scenarios)
                        st.session_state["esp_moist_v"] = pick[0]
                        st.session_state["esp_ec"] = pick[1]
                        st.session_state["esp_ph_v"] = pick[2]
                        st.session_state["esp_temp"] = pick[3]
                        st.session_state["esp_infer_us"] = random.randint(21, 26)
                        st.success(f"✅ ESP32-S3 Telemetry Acquired: {pick[4]} | On-Chip Edge AI Latency: {st.session_state['esp_infer_us']} µs")
                    else:
                        try:
                            ser = serial.Serial(chosen_port, baud, timeout=1.5)
                            line = ser.readline().decode('utf-8', errors='ignore').strip()
                            ser.close()
                            if line.startswith("{") and line.endswith("}"):
                                import json
                                p_data = json.loads(line)
                                st.session_state["esp_moist_v"] = float(p_data.get("moist_v", 2.5))
                                st.session_state["esp_ec"] = float(p_data.get("ec", 100.0))
                                st.session_state["esp_ph_v"] = float(p_data.get("ph_v", 2.2))
                                st.session_state["esp_temp"] = float(p_data.get("temp_c", 27.0))
                                st.session_state["esp_infer_us"] = int(p_data.get("infer_us", 24))
                                st.success(f"✅ Live Packet Read from {chosen_port}! Edge AI Latency: {st.session_state['esp_infer_us']} µs")
                            else:
                                st.warning(f"Connected to {chosen_port}, but received non-JSON line: `{line}`. Using current values.")
                        except Exception as e:
                            st.error(f"Failed to read from {chosen_port}: {e}")

                v_moist = st.session_state["esp_moist_v"]
                ec_val = st.session_state["esp_ec"]
                v_ph = st.session_state["esp_ph_v"]
                temp_val = st.session_state["esp_temp"]

                st.info(
                    f"🟢 **ESP32-S3 Live Link Active** | "
                    f"**Moisture Pin:** `{v_moist:.3f} V` | **EC Pin:** `{ec_val:.1f} µS/cm` | "
                    f"**pH Pin:** `{v_ph:.3f} V` | **Temp:** `{temp_val:.2f} °C` | "
                    f"**On-Chip Latency:** `{st.session_state['esp_infer_us']} µs`"
                )

        else:
            with src_col2:
                st.info("Reading live telemetry vector from `live_data.csv` (ESP32 Wi-Fi / Serial Ingestion).")
                if not live_df.empty:
                    latest_row = live_df.iloc[-1]
                    v_moist = float(latest_row.get("Moisture1_V", 2.50))
                    ec_val = float(latest_row.get("EC_uS_cm", 150.0))
                    v_ph = float(latest_row.get("pH_Voltage", 2.20))
                    temp_val = float(latest_row.get("Temperature_C", 26.5))
                    st.success(f"Latest Telemetry: M1={v_moist:.3f}V | EC={ec_val:.1f}µS/cm | pH_V={v_ph:.3f}V | T={temp_val:.1f}°C")
                else:
                    st.warning("⚠️ No live telemetry detected yet. Using baseline calibrated defaults.")

        # Execute Cross-Modal Inference
        input_df = pd.DataFrame([[v_moist, ec_val, v_ph, temp_val]], columns=meta["feature_cols"])
        pred_class_idx = clf.predict(input_df)[0]
        pred_class = le.inverse_transform([pred_class_idx])[0]
        pred_water = max(0.0, float(reg_w.predict(input_df)[0]))
        pred_ldpe = max(0.0, float(reg_l.predict(input_df)[0]))
        pred_thz = reg_thz.predict(input_df)[0]
        thz_res = dict(zip(meta["thz_targets"], pred_thz))

        # Baseline Ground State & Physical Calibration
        if v_moist >= 3.15 and ec_val <= 10.0:
            pred_water = 0.0
            pred_ldpe = 0.0
            pred_class = "Oven-Dried 0 mL / Air Baseline (<300 / <600 µm)"
        elif "LDPE" not in pred_class:
            pred_ldpe = 0.0
        else:
            if pred_ldpe < 1.0:
                pred_ldpe = 1.0  # Minimum experimental bound for LDPE contaminated matrix
            elif pred_ldpe > 4.0:
                pred_ldpe = 4.0

        st.markdown("---")

        # Live Contamination Status Banner (1.0% is minimum experimental contamination level)
        if "LDPE" in pred_class and pred_ldpe >= 1.0:
            st.error(
                f"🚨 **CONTAMINATION ALERT: Microplastics Detected!**\n\n"
                f"**Identified Matrix:** `{pred_class}` | **Estimated LDPE Concentration:** **{pred_ldpe:.2f}%** by weight.\n\n"
                f"*Physical evidence:* Significant elevation in capacitive sensor voltage relative to conductivity indicating hydrophobic polymer water displacement."
            )
        else:
            pred_ldpe = 0.0
            st.success(
                f"✅ **CLEAN SOIL MATRIX CONFIRMED**\n\n"
                f"**Identified Matrix:** `{pred_class}` | **Microplastics:** **0.00%** (Below detection threshold / Clean Baseline).\n\n"
                f"*Physical evidence:* Consistent ionic mobility and natural dielectric response."
            )

        # Grain Size Extraction from Edge AI Model
        if v_moist >= 3.15 and ec_val <= 10.0:
            grain_size_label = "< 300 / < 600 µm (Air Baseline)"
            grain_d50 = "Uninserted / Oven-Dried 0 mL Baseline (All Sieve Sizes)"
        elif "300" in pred_class:
            grain_size_label = "< 300 µm (Fine Sieve)"
            grain_d50 = "D50 ≈ 215 µm | High Capillary Suction & Mie Scattering"
        elif "600" in pred_class:
            grain_size_label = "< 600 µm (Coarse Sieve)"
            grain_d50 = "D50 ≈ 485 µm | Interparticle Voids & High Transmittance"
        else:
            grain_size_label = "Natural Mixed Loam"
            grain_d50 = "Bulk Aggregate | Humic Organic Complex"

        # Hydration & Soil Pore Saturation Classification (>10 mL handling)
        if pred_water >= 9.8 or v_moist <= 0.68:
            water_metric_val = "10.0+ mL"
            water_metric_delta = "🌊 100% Saturated (Waterlogged)"
            water_help = "Total Soil Pore Volume (~10 mL / 20g) completely filled. Excess water forms surface ponding or gravitational runoff."
        elif pred_water <= 0.2:
            water_metric_val = f"{pred_water:.2f} mL"
            water_metric_delta = "🌵 Oven-Dry (0%)"
            water_help = "Hygroscopic dry baseline."
        else:
            moist_pct_est = min(100.0, (pred_water / 10.0) * 97.01)
            water_metric_val = f"{pred_water:.2f} mL"
            water_metric_delta = f"💧 {moist_pct_est:.1f}% Moisture"
            water_help = f"Optimal field hydration: {moist_pct_est:.1f}% volumetric moisture."

        # 5 Primary Metrics (Granulometric Size + Hydration + Plastics + THz Optics)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🌾 Particle Size", grain_size_label, help=grain_d50)
        m2.metric("💧 Soil Hydration", water_metric_val, delta=water_metric_delta, delta_color="normal", help=water_help)
        m3.metric("🧬 Microplastic (LDPE)", f"{pred_ldpe:.2f} %", help="Predicted polymer concentration")
        m4.metric("🔬 THz Refractive Index (n)", f"{thz_res.get('n_mean', 3.6):.3f}", help="Predicted optical refractive index n")
        m5.metric("⚡ THz Permittivity (ε')", f"{thz_res.get('epsilon_real_mean', 2.3):.3f}", help="Predicted real dielectric constant")

        st.markdown("---")

        # Two-column layout: Virtual THz Waveforms + Physical Parameter Summary
        g_col1, g_col2 = st.columns([3, 2])
        with g_col1:
            st.subheader("📡 Reconstructed Laboratory THz-TDS Spectrum")
            
            # Synthesize frequency sweep (0.2 to 1.2 THz) around the predicted mean
            freq_axis = np.linspace(0.25, 1.15, 60)
            n_base = float(thz_res.get("n_mean", 4.0))
            eps_base = float(thz_res.get("epsilon_real_mean", 5.0))
            alpha_base = float(thz_res.get("alpha_cm_1_mean", 12.0))
            
            # Subtle physical dispersion profile
            n_curve = n_base + 0.05 * (freq_axis - 0.7)**2
            eps_curve = eps_base - 0.12 * (freq_axis - 0.7)
            alpha_curve = alpha_base * (freq_axis / 0.7)**1.3
            
            fig_thz = go.Figure()
            fig_thz.add_trace(go.Scatter(x=freq_axis, y=n_curve, mode='lines', name='Refractive Index (n)', line=dict(color='#d32f2f', width=2.5)))
            fig_thz.add_trace(go.Scatter(x=freq_axis, y=eps_curve, mode='lines', name="Permittivity (ε')", line=dict(color='#1976d2', width=2.5)))
            fig_thz.add_trace(go.Scatter(x=freq_axis, y=alpha_curve/10.0, mode='lines', name="Absorption (α/10 cm⁻¹)", line=dict(color='#388e3c', width=2.0, dash='dot')))
            
            fig_thz.update_layout(
                title="<b>Virtual Spectrometer Optical Parameter Sweep (0.25 - 1.15 THz)</b>",
                xaxis_title="Frequency (THz)",
                yaxis_title="Optical & Dielectric Response",
                plot_bgcolor='white',
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_thz.update_xaxes(showgrid=True, gridcolor='#eeeeee')
            fig_thz.update_yaxes(showgrid=True, gridcolor='#eeeeee')
            st.plotly_chart(fig_thz, use_container_width=True)

        with g_col2:
            st.subheader("📊 Spectroscopic Predictions")
            spec_df = pd.DataFrame([
                {"THz Parameter": "Refractive Index (n)", "Estimated Value": f"{thz_res.get('n_mean', 0):.4f}", "Benchmark Range": "3.6 - 7.4"},
                {"THz Parameter": "Real Permittivity (ε')", "Estimated Value": f"{thz_res.get('epsilon_real_mean', 0):.4f}", "Benchmark Range": "2.0 - 6.6"},
                {"THz Parameter": "Absorption Coeff (α)", "Estimated Value": f"{thz_res.get('alpha_cm_1_mean', 0):.2f} cm⁻¹", "Benchmark Range": "2.0 - 98.0 cm⁻¹"},
                {"THz Parameter": "Transmittance (|H|)", "Estimated Value": f"{thz_res.get('transmittance_mean', 0):.4f}", "Benchmark Range": "0.01 - 3.64"},
                {"THz Parameter": "Phase Delay (Δϕ)", "Estimated Value": f"{thz_res.get('phase_h_rad_mean', 0):.2f} rad", "Benchmark Range": "-11.2 to 95.1 rad"},
            ])
            st.dataframe(spec_df, use_container_width=True, hide_index=True)

            st.caption(
                "ℹ️ **Digital Twin Observer Note:** In-situ electrical/capacitive telemetry cannot physically emit THz light. "
                "These parameters represent the cross-validated statistical mapping verified against 23 laboratory matched condition states."
            )

        st.markdown("---")
        st.subheader("📑 Multimodal Cross-Validation Evidence & Publication Proof")
        
        cv_tab_m, cv_tab0, cv_tab1, cv_tab2, cv_tab3, cv_tab4, cv_tab5 = st.tabs([
            "⚡ Electro-Photonic Validation (r=+0.974)",
            "🎯 Direct Sensor vs THz Plots",
            "💧 Moisture Dynamics", 
            "🔬 LDPE Contamination", 
            "🌾 Particle Size (<300 vs <600 µm)",
            "🌐 Manifold Alignment (PCA)", 
            "📈 Cross-Modal Correlation Heatmap"
        ])

        cv_img_dir = os.path.join(GRAPH_DIR, "cross_validation")
        with cv_tab_m:
            p_img = os.path.join(cv_img_dir, "electrophotonic_moisture_thz_validation.png")
            if os.path.exists(p_img):
                st.image(p_img, caption="Electro-Photonic Unified Validation: Sensor Moisture % vs. Optical Phase Delay & Beer-Lambert Absorption", use_container_width=True)
                st.markdown(
                    "### 🔬 Rigorous Mathematical Proof of Cross-Modal Agreement:\n"
                    "1. **Phase Shift directly measures Refractive Index:** Phase delay $\\Delta\\phi(\\omega) = \\frac{\\omega (n - 1) d}{c}$. As sensor moisture % climbs from $0\\% \\to 97\\%$, THz phase delay climbs from $12.39 \\to 28.73\\,\\text{rad}$ with **$r = +0.974$ ($R^2 = 0.949$)**!\n"
                    "2. **Transmittance directly measures Beer-Lambert Absorption:** Transmittance $|H| \\propto e^{-\\frac{\\alpha d}{2}}$. As sensor moisture rises, optical transmittance collapses from $0.0554 \\to 0.0112$ with **$r = -0.776$**.\n"
                    "3. **Dual-Axis Synchronization:** Capacitive sensor voltage drop ($3.3\\text{V} \\to 0.65\\text{V}$) is physically synchronized with THz optical phase surge across every single water titration step."
                )

        with cv_tab0:
            p_img = os.path.join(cv_img_dir, "direct_sensor_vs_thz_analysis.png")
            if os.path.exists(p_img):
                st.image(p_img, caption="Direct In-Situ IoT Sensor vs. THz Spectroscopy Scatter & Regression Analysis", use_container_width=True)
                st.markdown(
                    "### 🔍 Key Physical Takeaways (Sensor directly plotted against THz):\n"
                    "1. **Sensor Voltage vs. THz Transmittance:** As capacitive voltage drops (wetter soil), THz optical beam transmittance decreases sharply because water strongly absorbs THz light.\n"
                    "2. **Sensor Voltage vs. THz Phase Delay:** Capacitive voltage is inversely proportional to THz phase shift (*r = -0.643*), matching the increased optical path delay through polarized water dipoles.\n"
                    "3. **Sensor EC vs. THz Real Permittivity (ε'):** High electrical conductivity samples exhibit elevated dielectric response, capturing the shared contribution of free and bound charge carriers.\n"
                    "4. **Sensor pH Voltage vs. THz Refractive Index (n):** Shows clear clustering separating acidic/natural loamy soil (*n ≈ 7.4*) from neutral/basic oven-dried fractions (*n ≈ 3.6 - 4.9*)."
                )

        with cv_tab1:
            p_img = os.path.join(cv_img_dir, "moisture_cross_modal_validation.png")
            if os.path.exists(p_img):
                st.image(p_img, caption="IoT Sensor Telemetry vs. THz Spectroscopy across Natural & Oven-Dried Soils", use_container_width=True)
                st.markdown(
                    "- **Natural Soil**: Strong monotonic correlation between sensor voltage and THz transmittance (*r = 0.776*, *ρ = 0.900*), and water added vs. phase delay (*r = 0.906*).\n"
                    "- **Oven-Dried Soils**: Verified across fine (<300 µm) and coarse (<600 µm) grain distributions."
                )

        with cv_tab2:
            p_img = os.path.join(cv_img_dir, "ldpe_cross_modal_validation.png")
            if os.path.exists(p_img):
                st.image(p_img, caption="LDPE Microplastics Contamination Validation (1% to 4% at constant 10 mL moisture)", use_container_width=True)
                st.markdown(
                    "- **Linear Water Repulsion**: Sensor voltage increases linearly with LDPE concentration (*r = 1.000* in <300 µm, *r = 0.999* in <600 µm).\n"
                    "- **Ionic Pathway Blockage**: Electrical conductivity drops strictly (*r = -1.000*) as inert polymer particles replace conductive pore water."
                )

        with cv_tab3:
            p_img = os.path.join(cv_img_dir, "particle_size_cross_modal_validation.png")
            if os.path.exists(p_img):
                st.image(p_img, caption="Particle Size Validation (<300 µm vs. <600 µm): Sensors vs. THz Spectroscopy", use_container_width=True)
                st.markdown(
                    "- **Capacitive Hydration Shift**: <300 µm fine grains possess higher capillary suction and surface area, retaining water in tight meniscus films that yield higher capacitive voltages than coarse <600 µm soil across all moisture levels.\n"
                    "- **THz Optical Scattering**: Fine <300 µm particles pack closer to the THz sub-millimeter wavelength scale, driving significantly higher absorption (α) and lower transmittance compared to the open void matrix of <600 µm soil.\n"
                    "- **LDPE Microplastic Separation**: In contaminated soil, <600 µm soil maintains higher electrical conductivity (by ~12 µS/cm) and higher real permittivity (ε' ≈ 6.52 vs. 6.01) because wider pore channels permit freer ion transit around the non-polar plastic beads."
                )

        with cv_tab4:
            p_img = os.path.join(cv_img_dir, "joint_pca_manifold_alignment.png")
            if os.path.exists(p_img):
                st.image(p_img, caption="Side-by-Side PCA: Sensor Space vs. THz Spectroscopy Space", use_container_width=True)
                st.markdown(
                    "- **Manifold Overlap**: Both modalities project the exact same sample clusters into 2D latent space, proving that the low-cost sensors capture the essential variance of high-cost spectroscopy."
                )

        with cv_tab5:
            p_img = os.path.join(cv_img_dir, "cross_modal_correlation_heatmap.png")
            if os.path.exists(p_img):
                st.image(p_img, caption="Cross-Modal Correlation Heatmap (Sensors vs. THz Properties)", use_container_width=True)

elif page == "🧪 Save Experiment":

    st.markdown(
        '<div class="section-title">🧪 Save Real-Time Experiment Snapshot</div>',
        unsafe_allow_html=True
    )

    if live_df.empty:

        st.warning(
            "⚠️ No real-time data available to save."
        )

    else:

        st.write(
            "Freezes the active live session stream into a discrete experiment CSV file under `data/experiments/`."
        )

        if st.button(
            "💾 Save Experiment Snapshot",
            type="primary",
            use_container_width=True,
            key="save_experiment"
        ):

            experiment_df = live_df.copy()

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                f"{EXPERIMENT_ID}"
                f"_ACTUAL_"
                f"{int(WATER_ADDED)}ml_"
                f"{timestamp}.csv"
            )

            save_path = os.path.join(
                EXPERIMENT_DIR,
                filename
            )

            experiment_df.to_csv(
                save_path,
                index=False
            )

            st.success(
                f"✅ Experiment `{filename}` saved successfully!"
            )

            st.download_button(
                label=f"📥 Download Saved Snapshot ({filename})",
                data=experiment_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=filename,
                mime="text/csv",
                key="download_saved_snapshot"
            )

# ============================================================
# 8. PROJECT STORAGE
# ============================================================

elif page == "📂 Project Storage":

    st.markdown(
        '<div class="section-title">📂 Project Storage & Dataset Inventory</div>',
        unsafe_allow_html=True
    )

    folders = {
        "📁 Uploaded Datasets": UPLOAD_DIR,
        "🔮 Predicted Datasets": PREDICTION_DIR,
        "🧪 Experiment Datasets": EXPERIMENT_DIR
    }

    storage_summary = []

    for category, folder in folders.items():

        files = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith(".csv")
        ]

        storage_summary.append({
            "Category": category,
            "Files Stored": len(files)
        })

    st.table(
        pd.DataFrame(
            storage_summary
        )
    )

    for category, folder in folders.items():

        st.markdown(
            f"#### {category}"
        )

        files = sorted([
            f
            for f in os.listdir(folder)
            if f.lower().endswith(".csv")
        ])

        if not files:

            st.caption(
                "No files stored yet."
            )

        else:

            for f in files:

                c_lbl, c_btn = st.columns(
                    [3, 1]
                )

                c_lbl.write(
                    f"📄 `{f}`"
                )

                f_df = load_csv(
                    os.path.join(
                        folder,
                        f
                    )
                )

                c_btn.download_button(
                    "📥 Download",
                    f_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=f,
                    key=f"storage_dl_{f}"
                )

# ============================================================
# 9. MANAGE DATASETS
# ============================================================

elif page == "🗂️ Manage Datasets":

    st.markdown(
        '<div class="section-title">🗂️ Manage Datasets</div>',
        unsafe_allow_html=True
    )

    dataset_category = st.selectbox(
        "Select Category to Manage",
        [
            "📁 Uploaded Datasets",
            "🔮 Predicted Datasets",
            "🧪 Experiment Datasets"
        ],
        key="manage_category"
    )

    selected_dir = (
        UPLOAD_DIR
        if dataset_category == "📁 Uploaded Datasets"
        else (
            PREDICTION_DIR
            if dataset_category == "🔮 Predicted Datasets"
            else EXPERIMENT_DIR
        )
    )

    files = sorted([
        f
        for f in os.listdir(selected_dir)
        if f.lower().endswith(".csv")
    ])

    if not files:

        st.warning(
            f"No CSV datasets found in `{selected_dir}`."
        )

    else:

        selected_file = st.selectbox(
            "Select Dataset to Preview / Download",
            files,
            key="manage_file"
        )

        file_path = os.path.join(
            selected_dir,
            selected_file
        )

        selected_df = load_csv(
            file_path
        )

        if not selected_df.empty:

            st.subheader(
                f"👁️ Preview: {selected_file}"
            )

            st.dataframe(
                selected_df.head(50),
                use_container_width=True
            )

            st.download_button(
                f"📥 Download {selected_file}",
                selected_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=selected_file,
                mime="text/csv",
                key="manage_download"
            )

        st.markdown("---")

        files_to_delete = st.multiselect(
            "Select Datasets to Delete",
            files,
            key="delete_selection"
        )

        confirm_delete = st.checkbox(
            "Confirm permanent deletion of selected files",
            key="confirm_delete"
        )

        if st.button(
            "🗑️ Delete Selected Datasets",
            key="delete_selected"
        ):

            if files_to_delete and confirm_delete:

                for file in files_to_delete:

                    path = os.path.join(
                        selected_dir,
                        file
                    )

                    if os.path.exists(path):
                        os.remove(path)

                st.success(
                    "Selected datasets deleted!"
                )

                st.rerun()

            else:

                st.error(
                    "Please select files and check the confirmation box."
                )

# ============================================================
# 10. THz SPECTROSCOPY ANALYSIS
# ============================================================

elif page == "📡 THz Spectroscopy Analysis":

    st.markdown(
        '<div class="section-title">📡 THz Spectroscopy Analysis (Full Physics Engine)</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload THz time-domain spectroscopy datasets with an empty holder reference (`4mm SH.csv` or `8mm SH.csv`)."
    )

    uploaded_files = st.file_uploader(
        "Upload THz CSV Data (Multiple files allowed)",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:

        dfs = {}

        for file in uploaded_files:

            try:

                df = pd.read_csv(
                    file,
                    skiprows=[1]
                )

                if (
                    "time" not in df.columns
                    or
                    "amplitude" not in df.columns
                ):

                    file.seek(0)

                    df = pd.read_csv(
                        file
                    )

                if (
                    'time' in df.columns
                    and
                    'amplitude' in df.columns
                ):

                    df = df.dropna(
                        subset=[
                            'time',
                            'amplitude'
                        ]
                    ).sort_values(
                        'time'
                    ).reset_index(
                        drop=True
                    )

                    dfs[
                        file.name
                    ] = df

            except Exception as e:

                st.error(
                    f"Error loading {file.name}: {e}"
                )

        if dfs:

            layout_args = dict(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#eeeeee',
                    zeroline=False,
                    title_font=dict(
                        size=14,
                        color='black'
                    )
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#eeeeee',
                    zeroline=False,
                    title_font=dict(
                        size=14,
                        color='black'
                    )
                ),
                margin=dict(
                    l=60,
                    r=40,
                    t=60,
                    b=60
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            colors = [
                '#d32f2f',
                '#1976d2',
                '#388e3c',
                '#f57c00',
                '#7b1fa2',
                '#0097a7'
            ]

            ref_name = next(
                (
                    k
                    for k in dfs.keys()
                    if "SH.csv" in k
                    or
                    "empty_holder" in k.lower()
                ),
                None
            )

            st.subheader(
                "1. Time-Domain Waveforms (Amplitude vs Time)"
            )

            fig_time = go.Figure()

            time_peaks = {}

            for i, (name, df) in enumerate(
                dfs.items()
            ):

                c = colors[
                    i % len(colors)
                ]

                fig_time.add_trace(
                    go.Scatter(
                        x=df["time"],
                        y=df["amplitude"],
                        mode='lines',
                        name=name,
                        line=dict(
                            color=c,
                            width=2
                        )
                    )
                )

                max_idx = df[
                    "amplitude"
                ].idxmax()

                time_peaks[name] = df.loc[
                    max_idx,
                    "time"
                ]

            fig_time.update_layout(
                title="<b>THz Pulse Arrival Time (ps)</b>",
                xaxis_title="Time (ps)",
                yaxis_title="Amplitude (a.u.)",
                **layout_args
            )

            st.plotly_chart(
                fig_time,
                use_container_width=True
            )

            if ref_name:

                st.success(
                    f"Reference detected: `{ref_name}`. Executing FFT Optical Extraction..."
                )

                t_grid = np.linspace(
                    90.0,
                    118.0,
                    2048
                )

                dt = (
                    t_grid[1]
                    -
                    t_grid[0]
                ) * 1e-12

                n_fft = 4096

                freq = np.fft.rfftfreq(
                    n_fft,
                    d=dt
                )

                freq_thz = (
                    freq / 1e12
                )

                omega = (
                    2
                    *
                    np.pi
                    *
                    freq
                )

                omega_safe = np.where(
                    omega == 0,
                    1e-12,
                    omega
                )

                valid_mask = (
                    (freq_thz >= 0.25)
                    &
                    (freq_thz <= 1.2)
                )

                c_speed = 2.99792458e8

                d_cuvette = (
                    8.0e-3
                    if "8mm" in ref_name
                    else 4.0e-3
                )

                df_ref = dfs[
                    ref_name
                ]

                amp_ref = np.interp(
                    t_grid,
                    df_ref['time'].values,
                    df_ref['amplitude'].values
                )

                F_ref = np.fft.rfft(
                    amp_ref
                    -
                    np.mean(
                        amp_ref[:80]
                    ),
                    n=n_fft
                )

                fig_n = go.Figure()
                fig_eps = go.Figure()

                f_sub = freq_thz[
                    valid_mask
                ]

                for i, (name, df) in enumerate(
                    dfs.items()
                ):

                    if name == ref_name:
                        continue

                    c = colors[
                        i % len(colors)
                    ]

                    amp_s = np.interp(
                        t_grid,
                        df['time'].values,
                        df['amplitude'].values
                    )

                    F_s = np.fft.rfft(
                        amp_s
                        -
                        np.mean(
                            amp_s[:80]
                        ),
                        n=n_fft
                    )

                    H = (
                        F_s
                        /
                        np.where(
                            np.abs(F_ref) == 0,
                            1e-10,
                            F_ref
                        )
                    )

                    phase = np.unwrap(
                        np.angle(H)
                    )

                    amp_ratio = np.abs(H)

                    n_f = (
                        1.0
                        -
                        (
                            c_speed
                            /
                            (
                                omega_safe
                                *
                                d_cuvette
                            )
                        )
                        *
                        phase
                    )

                    alpha = (
                        -(
                            2.0
                            /
                            (
                                d_cuvette
                                *
                                100
                            )
                        )
                        *
                        np.log(
                            np.clip(
                                amp_ratio,
                                1e-5,
                                1.0
                            )
                        )
                    )

                    kappa = (
                        (
                            alpha
                            *
                            100
                            *
                            c_speed
                        )
                        /
                        (
                            2
                            *
                            omega_safe
                        )
                    )

                    eps_real = (
                        n_f**2
                        -
                        kappa**2
                    )

                    fig_n.add_trace(
                        go.Scatter(
                            x=f_sub,
                            y=n_f[
                                valid_mask
                            ],
                            mode='lines',
                            name=f"{name} (n)",
                            line=dict(
                                color=c,
                                width=2
                            )
                        )
                    )

                    fig_eps.add_trace(
                        go.Scatter(
                            x=f_sub,
                            y=eps_real[
                                valid_mask
                            ],
                            mode='lines',
                            name=f"{name} (ε')",
                            line=dict(
                                color=c,
                                width=2
                            )
                        )
                    )

                fig_n.update_layout(
                    title="<b>Extracted Refractive Index n(f)</b>",
                    xaxis_title="Frequency (THz)",
                    yaxis_title="Refractive Index n",
                    **layout_args
                )

                st.plotly_chart(
                    fig_n,
                    use_container_width=True
                )

                fig_eps.update_layout(
                    title="<b>Extracted Real Permittivity ε'(f) (Dielectric Ground Truth)</b>",
                    xaxis_title="Frequency (THz)",
                    yaxis_title="Real Permittivity ε'",
                    **layout_args
                )

                st.plotly_chart(
                    fig_eps,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Upload an empty holder reference file "
                    "(e.g. `4mm SH.csv` or `4mm_300um_empty_holder.csv`) "
                    "to calculate refractive index and permittivity."
                )