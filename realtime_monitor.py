import psutil
import joblib
import pandas as pd
import time
from datetime import datetime
import os

model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Real-time System Health Monitoring Started...")

boot_time = psutil.boot_time()

log_file = "anomaly_log.csv"

if not os.path.exists(log_file):
    df = pd.DataFrame(columns=[
        "timestamp",
        "cpu_usage",
        "ram_usage",
        "disk_usage",
        "net_sent",
        "net_recv",
        "process_count",
        "uptime_seconds"
    ])
    df.to_csv(log_file, index=False)

while True:

    timestamp = datetime.now()

    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent

    net = psutil.net_io_counters()
    net_sent = net.bytes_sent
    net_recv = net.bytes_recv

    processes = len(psutil.pids())
    uptime = time.time() - boot_time

    data = pd.DataFrame([[
        cpu,
        ram,
        disk,
        net_sent,
        net_recv,
        processes,
        uptime
    ]], columns=[
        "cpu_usage",
        "ram_usage",
        "disk_usage",
        "net_sent",
        "net_recv",
        "process_count",
        "uptime_seconds"
    ])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    if prediction[0] == -1:
        print("⚠️  ANOMALY DETECTED!")

        row = pd.DataFrame([[
            timestamp,
            cpu,
            ram,
            disk,
            net_sent,
            net_recv,
            processes,
            uptime
        ]], columns=[
            "timestamp",
            "cpu_usage",
            "ram_usage",
            "disk_usage",
            "net_sent",
            "net_recv",
            "process_count",
            "uptime_seconds"
        ])

        row.to_csv(log_file, mode='a', header=False, index=False)

    else:
        print("✅ System Normal")

    print(f"CPU: {cpu}% | RAM: {ram}% | Processes: {processes}")

    time.sleep(5)
