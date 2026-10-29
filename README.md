# Multimodal Soil Spectrometer Digital Twin

### THz-TDS + Embedded Soil Sensing + TinyML on ESP32-S3

A research project connecting low-cost embedded soil measurements with laboratory-grade **Terahertz Time-Domain Spectroscopy (THz-TDS)** to study soil moisture response and investigate electronic and optical signatures associated with LDPE contamination.

The system integrates controlled soil preparation, THz characterization, embedded sensing, machine-learning inference, and an interactive Streamlit Digital Twin.

## Overview

Conventional soil probes are useful for continuous monitoring, but their readings can be influenced by soil texture, packing, temperature, dissolved ions, and sensor contact. THz-TDS provides a complementary laboratory technique for analyzing electromagnetic pulse transmission, phase response, and derived optical properties.

This project investigates whether embedded electronic measurements can be associated with laboratory-derived THz properties and whether a compact machine-learning model can estimate selected outputs locally on an ESP32-S3.

## System Architecture

- **Physical sample preparation:** Natural loamy soil, oven-dried and sieved soil fractions, and LDPE-mixed soil.
- **Embedded sensing:** ESP32-S3 with capacitive moisture, electrical conductivity (EC), pH-electrode potential, and temperature inputs.
- **Laboratory characterization:** Transmission-mode THz-TDS using nominal 4 mm and 8 mm sample holders.
- **Signal processing:** Waveform analysis and extraction of optical and dielectric parameters.
- **Machine learning:** Model training and Leave-One-Group-Out Cross-Validation (LOGO-CV).
- **Edge deployment:** Compact model inference on the ESP32-S3.
- **Digital Twin:** Streamlit dashboard for telemetry, predictions, and visualization.

## Research Objectives

- Investigate how water additions affect embedded sensor responses and THz-TDS measurements.
- Compare optical responses across sample path lengths and soil particle-size fractions.
- Investigate electronic and optical signatures associated with LDPE-mixed soil.
- Develop a condition-matched dataset linking embedded measurements with laboratory optical measurements.
- Evaluate model generalization using LOGO-CV.
- Explore low-latency, local inference on the ESP32-S3.
- Integrate sensor telemetry and model outputs into an interactive dashboard.

## Experimental Design

The project report describes **23 condition-matched soil states** across three experimental groups.

| Experimental group | Conditions | Purpose |
|---|---|---|
| Natural loamy soil | Water additions of 0, 2.5, 5, 7.5, and 10 mL to fixed 20 g soil aliquots | Study moisture-related sensor and THz responses |
| Oven-dried, sieved soil | Fine and coarser fractions reported as `<300 µm` and `<600 µm` | Investigate particle-size effects |
| LDPE-mixed soil | Reported LDPE contamination levels from 1% to 4% | Investigate electronic and optical responses associated with polymer addition |

The reported THz-TDS frequency band is **0.2–1.2 THz**. Measurements were performed in transmission mode using nominal 4 mm and 8 mm sample holders.

*Note: Added-water volume is an experimental preparation variable and should not automatically be interpreted as gravimetric or volumetric soil moisture content.*

## Embedded Sensing Subsystem

The ESP32-S3 node acquires electrical measurements that complement the laboratory THz-TDS data.

| Sensor input | Purpose |
|---|---|
| Capacitive moisture sensor | Measures an electrical response associated with soil moisture |
| EC/TDS sensor | Measures electrical conductivity and ionic response |
| pH electrode interface | Records electrode potential |
| Temperature sensor | Records substrate temperature for contextual analysis and compensation |

The pH electrode output is treated as an electrode-potential measurement unless calibration with suitable buffer solutions establishes a valid pH reading.

## THz-TDS Signal Processing

The optical analysis uses time-domain waveforms and frequency-domain measurements to characterize the sample response.

Parameters analyzed include:

- Reference and sample field magnitude
- Transfer-function magnitude and phase
- Transmittance
- Refractive index (`n`)
- Extinction coefficient (`k`)
- Real and imaginary permittivity (`ε′`, `ε″`)
- Absorption coefficient (`α`)
- Phase delay and waveform timing features

The processing workflow considers sample path length, reference measurements, phase unwrapping, attenuation, signal quality, and the usable frequency range.

## Machine Learning and Virtual Spectrometer

The project explores a **virtual spectrometer** that estimates selected THz-derived properties or sample-condition outputs from embedded sensor measurements.

Tree-based machine-learning models are evaluated using **Leave-One-Group-Out Cross-Validation (LOGO-CV)**. This approach helps reduce the risk of overly optimistic results caused by placing closely related experimental observations in both training and test sets.

The model is intended to approximate selected laboratory-derived outputs from embedded inputs. It does not physically generate THz radiation or replace a laboratory THz-TDS instrument.

## TinyML and ESP32-S3 Deployment

The project includes a deployment pathway for executing model inference locally on the ESP32-S3 using an embedded-compatible representation of the trained model.

The thesis reports an inference latency below 25 µs. Actual performance depends on the model, firmware build, processor configuration, and timing methodology.

## Digital Twin Dashboard

The Streamlit dashboard connects embedded telemetry with an interactive visualization layer.

Reported dashboard functionality includes:

- Live sensor telemetry
- Inferred sample condition and water-addition output
- LDPE-related model output
- Visualization of estimated optical properties
- PCA and manifold visualizations
- Wi-Fi/HTTP or serial telemetry integration

Model predictions are estimates and should be distinguished from direct sensor readings and laboratory measurements.

## Reported Evaluation

The project thesis reports the following results:

| Evaluation metric | Reported result |
|---|---:|
| Water-related prediction | `R² = 0.787` under LOGO-CV |
| LDPE-related prediction | `R² = 0.878` under LOGO-CV |
| ESP32-S3 inference latency | `<25 µs` |

These values are reported from the project evaluation and apply to the specific dataset, target definitions, and validation methodology used. They are not guarantees of performance on other soil types or field conditions.

## Technology Stack

- **Embedded hardware:** ESP32-S3 and analog sensor interfaces
- **Optical characterization:** Terahertz Time-Domain Spectroscopy
- **Data processing:** Python, NumPy, pandas, and scientific plotting tools
- **Machine learning:** Tree-based modelling and group-aware validation
- **Edge deployment:** Embedded-compatible model representation and inference
- **Dashboard:** Streamlit, Plotly, and telemetry integration
- **Communication:** Wi-Fi/HTTP and serial interfaces

## Repository Structure

The following is a suggested organization. Adjust it to match the files actually included in your repository.

```text
.
├── firmware/
│   └── ESP32-S3 sensor acquisition and inference code
├── data/
│   └── Experimental dataset and condition-matched measurements
├── analysis/
│   └── THz waveform and spectrum processing
├── models/
│   └── Training, validation, and exported model files
├── dashboard/
│   └── Streamlit Digital Twin application
├── docs/
│   └── Report, hardware documentation, and diagrams
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Ensure the dependency list matches the Python packages imported by the actual project files.

### 4. Run the dashboard

If the Streamlit entry point is named `app.py`:

```bash
streamlit run app.py
```

Configure the ESP32-S3 IP address or serial connection according to the firmware and dashboard settings.

### 5. Flash the embedded firmware

Open the firmware in the development environment used by the project, select the correct ESP32-S3 board, verify sensor pin mappings, configure local network credentials, and build and flash the firmware.

**Do not commit Wi-Fi passwords, API keys, or other secrets to GitHub.**

## Limitations

- The tested soil type and preparation conditions limit generalization to other soils.
- The LDPE experiments do not establish equivalent performance for all microplastic types.
- Embedded sensor readings may depend on calibration, soil packing, temperature, salinity, and sensor contact.
- pH-electrode potential is not equivalent to calibrated pH without appropriate calibration.
- THz-derived optical parameters depend on sample geometry, reference measurements, phase processing, and signal quality.
- The size of the experimental dataset limits conclusions about generalization to unseen conditions.
- A virtual spectrometer and Digital Twin do not replace independent laboratory verification.
- Reported inference latency depends on the deployed model and firmware configuration.


## License

Choose and add a suitable `LICENSE` file before permitting reuse of the code or documentation. Confirm that you have permission to publish any third-party datasets, instrument exports, figures, or reference materials.
