# -*- coding: utf-8 -*-
"""
=============================================================================
ESP32-S3 AUTONOMOUS SoftAP WI-FI BRIDGE (OFFLINE / NO INTERNET REQUIRED)
=============================================================================
Polls http://192.168.4.1/api/data over the ESP32-S3's private Wi-Fi access point
and logs telemetry directly into dashboard/csv/live_data.csv in real time.
=============================================================================
"""

import sys, time, os, csv
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import requests
except ImportError:
    print("[ERROR] requests library missing. Run: pip install requests")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv") if "dashboard" in BASE_DIR else os.path.join(r"C:\Users\priya\OneDrive\Desktop\ECM\CAPSTONE\SOIL_DIGITAL_TWIN\dashboard", "csv")
os.makedirs(CSV_DIR, exist_ok=True)
filename = os.path.join(CSV_DIR, "live_data.csv")

HEADERS = [
    "Timestamp", "Experiment_ID", "Soil_Type", "Water_Added_ml",
    "Temperature_C", "Temperature_F", "Temperature_K",
    "Moisture1_Raw", "Moisture1_V",
    "Moisture2_Raw", "Moisture2_V",
    "Moisture_Avg", "Moisture_Diff", "Moisture_Spread", "Uniformity",
    "pH_Raw", "pH_Voltage",
    "Nitrogen_mg_kg", "Phosphorus_mg_kg", "Potassium_mg_kg",
    "EC_uS_cm"
]

if not os.path.exists(filename) or os.path.getsize(filename) == 0:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADERS)

URL = "http://192.168.4.1/api/data"

print("\n=======================================================")
print("  ESP32-S3 AUTONOMOUS SoftAP WI-FI BRIDGE (OFFLINE)")
print("=======================================================")
print(f"[*] Target Endpoint: {URL}")
print(f"[*] Target CSV File: {filename}")
print("[*] Note: Ensure your laptop is connected to Wi-Fi: SOIL_DIAGNOSTIC_NODE")
print("=======================================================\n")

consecutive_errors = 0

try:
    while True:
        try:
            res = requests.get(URL, timeout=1.5)
            if res.status_code == 200:
                d = res.json()
                consecutive_errors = 0
                
                temp_c = float(d.get("temp_c", 27.5))
                temp_f = (temp_c * 9.0 / 5.0) + 32.0
                temp_k = temp_c + 273.15
                
                m1_raw = int(d.get("m1_raw", 2500))
                m1_v = float(d.get("m1_v", 2.0))
                m2_raw = int(d.get("m2_raw", 2500))
                m2_v = float(d.get("m2_v", 2.0))
                
                m_diff = float(d.get("m_diff", 0.0))
                m_avg = (m1_raw + m2_raw) / 2.0
                uniformity = float(d.get("uniformity", 1.0))
                
                ph_raw = int(d.get("ph_raw", 2300))
                ph_v = float(d.get("ph_v", 1.85))
                
                ec_comp = float(d.get("ec_comp", 150.0))
                
                regime = d.get("regime", "Natural Loamy")
                water_ml = float(d.get("water_ml", 0.0))
                ldpe_pct = float(d.get("ldpe_pct", 0.0))
                is_plastic = d.get("plastic", 0)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                row = [
                    timestamp,
                    "LIVE_ESP32",
                    regime,
                    f"{water_ml:.1f}",
                    f"{temp_c:.2f}",
                    f"{temp_f:.2f}",
                    f"{temp_k:.2f}",
                    m1_raw,
                    f"{m1_v:.3f}",
                    m2_raw,
                    f"{m2_v:.3f}",
                    f"{m_avg:.1f}",
                    f"{m_diff:.1f}",
                    f"{m_diff:.1f}",
                    f"{uniformity:.3f}",
                    ph_raw,
                    f"{ph_v:.3f}",
                    "0.0", "0.0", "0.0",
                    f"{ec_comp:.1f}"
                ]
                
                with open(filename, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                
                status_icon = "🚨 PLASTIC ALERT!" if is_plastic else "🌱 CLEAN"
                print(f"[{timestamp}] M1={m1_v:.3f}V | EC={ec_comp:.1f}uS | pH={ph_v:.3f}V | T={temp_c:.1f}C | AI: {regime} (LDPE: {ldpe_pct:.2f}%) {status_icon}")
                
            else:
                print(f"[!] HTTP Error: {res.status_code}")
                
        except requests.exceptions.RequestException:
            consecutive_errors += 1
            if consecutive_errors == 1:
                print("[!] Waiting for ESP32-S3... (Connect your laptop Wi-Fi to 'SOIL_DIAGNOSTIC_NODE')")
        
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\n[INFO] Stopped Wi-Fi bridge.")
