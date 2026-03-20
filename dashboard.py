import streamlit as st
import psutil
import joblib
import pandas as pd
import time

anomaly_model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

rul_model = joblib.load("rul_model.pkl")
rul_scaler = joblib.load("rul_scaler.pkl")

st.set_page_config(page_title="System Health Monitor", layout="wide")

st.title("AI System Health Monitoring Dashboard")

col1, col2 = st.columns(2)

cpu_placeholder = col1.empty()
ram_placeholder = col2.empty()

status_placeholder = st.empty()
rul_placeholder = st.empty()

chart_placeholder = st.empty()

chart_data = pd.DataFrame(columns=["CPU Usage", "RAM Usage"])

boot_time = psutil.boot_time()

while True:

    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent

    net = psutil.net_io_counters()
    net_sent = net.bytes_sent
    net_recv = net.bytes_recv

    processes = len(psutil.pids())
    uptime = time.time() - boot_time

    data = pd.DataFrame([[
        cpu, ram, disk,
        net_sent, net_recv,
        processes, uptime
    ]], columns=[
        "cpu_usage", "ram_usage", "disk_usage",
        "net_sent", "net_recv",
        "process_count", "uptime_seconds"
    ])

    scaled_data = scaler.transform(data)
    anomaly_pred = anomaly_model.predict(scaled_data)

    if anomaly_pred[0] == -1:
        anomaly_status = "ANOMALY DETECTED"
    else:
        anomaly_status = "System Normal"

    rul_scaled = rul_scaler.transform(data)
    rul_steps = rul_model.predict(rul_scaled)[0]

    rul_minutes = (rul_steps * 5) / 60

    if rul_minutes < 2:
        rul_status = "CRITICAL"
    elif rul_minutes < 5:
        rul_status = "Moderate Risk"
    else:
        rul_status = "Stable"

    cpu_placeholder.metric("CPU Usage (%)", cpu)
    ram_placeholder.metric("RAM Usage (%)", ram)

    status_placeholder.subheader(f"Status: {anomaly_status}")
    rul_placeholder.subheader(
    f"⏳ Time to Anomaly: {rul_minutes:.2f} min | {rul_status}"
    )
    new_row = pd.DataFrame([[cpu, ram]], columns=["CPU Usage", "RAM Usage"])
    chart_data = pd.concat([chart_data, new_row]).tail(50)

    chart_placeholder.line_chart(chart_data)

    time.sleep(2)
