# -*- coding: utf-8 -*-
"""
==============================================================================
ESP32-S3 SERIAL BRIDGE & LIVE TELEMETRY TESTER
==============================================================================
Connects to an ESP32-S3 over USB Serial, parses real-time Edge AI predictions,
and demonstrates live inference for Soil Regimes, Microplastics, and Virtual THz.
==============================================================================
"""

import sys, time, json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)

def list_ports():
    ports = serial.tools.list_ports.comports()
    print("\n--- AVAILABLE SERIAL PORTS ---")
    usb_ports = []
    if not ports:
        print("  No serial ports detected! Plug in your ESP32-S3 via USB-C.")
    for p in ports:
        is_bt = "Bluetooth" in p.description
        tag = "[BLUETOOTH]" if is_bt else "[USB/UART]"
        print(f"  {tag} [{p.device}] - {p.description}")
        if not is_bt:
            usb_ports.append(p.device)
    print("------------------------------\n")
    return usb_ports, ports

def run_bridge(port_name=None, baudrate=115200):
    usb_ports, all_ports = list_ports()
    
    if not port_name:
        if usb_ports:
            port_name = usb_ports[0]
            print(f"[AUTO] Selecting USB UART port: {port_name}")
        else:
            print("[INFO] No dedicated USB-UART port detected. Running SIMULATED VIRTUAL ESP32-S3 mode...")
            simulate_stream()
            return

    try:
        ser = serial.Serial(port_name, baudrate, timeout=1.0)
        print(f"[SUCCESS] Connected to ESP32-S3 on {port_name} @ {baudrate} baud.")
        print("[INFO] Waiting for live Edge AI telemetry stream...\n")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    print_telemetry(data)
                except json.JSONDecodeError:
                    print("[RAW]", line)
            else:
                print("[ESP32-S3 LOG]", line)
                
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"[ERROR] Serial connection failed: {e}")
        print("[INFO] Falling back to simulated stream demonstration...")
        simulate_stream()

def print_telemetry(d):
    print("=" * 70)
    print(f"  ESP32-S3 EDGE AI LIVE TELEMETRY  |  Latency: {d.get('infer_us', 24)} µs")
    print("=" * 70)
    print(f"  [INPUTS]  Moisture: {d.get('moist_v', 0):.3f} V | EC: {d.get('ec', 0):.1f} µS/cm | pH: {d.get('ph_v', 0):.3f} V | Temp: {d.get('temp_c', 0):.2f} °C")
    print(f"  [REGIME]  {d.get('regime', 'Unknown')} (ID: {d.get('regime_id', 0)})")
    size_str = d.get('particle_size', '< 300 um (Fine)' if '300' in d.get('regime', '') else ('< 600 um (Coarse)' if '600' in d.get('regime', '') else 'Natural Mixed Loam'))
    print(f"  [GRAIN]   Soil Particle Size: {size_str} (Cutoff: {d.get('particle_um', 300 if '300' in size_str else (600 if '600' in size_str else 0))} µm)")
    print(f"  [WATER]   Added: {d.get('water_ml', 0):.2f} mL  -->  Estimated Moisture: {d.get('moist_pct', 0):.1f} %")
    
    ldpe = d.get('ldpe_pct', 0)
    plastic_status = "[ALERT] CONTAMINATED!" if d.get('plastic', 0) else "[CLEAN] No Plastic"
    print(f"  [PLASTIC] LDPE Concentration: {ldpe:.2f} %  -->  {plastic_status}")
    
    print("  [VIRTUAL THz SPECTROSCOPY]")
    print(f"    - Refractive Index (n):        {d.get('thz_n', 0):.3f}")
    print(f"    - Real Permittivity (ε'):      {d.get('thz_eps', 0):.2f}")
    print(f"    - Absorption Coeff (α):        {d.get('thz_alpha', 0):.2f} cm⁻¹")
    print(f"    - Optical Transmittance (|H|): {d.get('thz_trans', 0):.4f}")
    print(f"    - Pulse Phase Delay (Δϕ):      {d.get('thz_phase', 0):.2f} rad")
    print("-" * 70 + "\n")

def simulate_stream():
    samples = [
        {"device": "ESP32-S3", "moist_v": 3.300, "ec": 0.0, "ph_v": 1.860, "temp_c": 28.62, "regime_id": 0, "regime": "Natural Loamy Soil", "water_ml": 0.0, "moist_pct": 0.0, "ldpe_pct": 0.0, "plastic": 0, "thz_n": 7.37, "thz_eps": 11.23, "thz_alpha": 4.12, "thz_trans": 0.0554, "thz_phase": 12.39, "infer_us": 22},
        {"device": "ESP32-S3", "moist_v": 1.350, "ec": 834.7, "ph_v": 1.810, "temp_c": 27.91, "regime_id": 0, "regime": "Natural Loamy Soil", "water_ml": 5.0, "moist_pct": 67.0, "ldpe_pct": 0.0, "plastic": 0, "thz_n": 7.35, "thz_eps": 12.98, "thz_alpha": 12.45, "thz_trans": 0.0215, "thz_phase": 27.19, "infer_us": 24},
        {"device": "ESP32-S3", "moist_v": 0.656, "ec": 2164.8, "ph_v": 1.750, "temp_c": 27.47, "regime_id": 0, "regime": "Natural Loamy Soil", "water_ml": 10.0, "moist_pct": 97.0, "ldpe_pct": 0.0, "plastic": 0, "thz_n": 7.36, "thz_eps": 13.52, "thz_alpha": 18.02, "thz_trans": 0.0112, "thz_phase": 27.83, "infer_us": 23},
        {"device": "ESP32-S3", "moist_v": 0.887, "ec": 54.3, "ph_v": 2.228, "temp_c": 28.71, "regime_id": 3, "regime": "LDPE Contaminated Soil (<300 um)", "water_ml": 10.0, "moist_pct": 97.0, "ldpe_pct": 4.0, "plastic": 1, "thz_n": 4.89, "thz_eps": 6.01, "thz_alpha": 2.15, "thz_trans": 0.0410, "thz_phase": 95.12, "infer_us": 25}
    ]
    
    print("[SIMULATOR] Demonstrating 4 consecutive live ESP32-S3 Edge AI predictions:\n")
    for s in samples:
        time.sleep(0.5)
        print_telemetry(s)

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else None
    run_bridge(port)
