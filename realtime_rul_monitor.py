import psutil
import joblib
import pandas as pd
import time

model = joblib.load("rul_model.pkl")
scaler = joblib.load("rul_scaler.pkl")

print("⏳ Real-time RUL Monitoring Started...")

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

    rul_steps = model.predict(scaled_data)[0]

    interval = 5  
    rul_seconds = rul_steps * interval
    rul_minutes = rul_seconds / 60

    if rul_minutes < 2:
        status = "HIGH RISK"
    elif rul_minutes < 5:
        status = "Moderate Risk"
    else:
        status = "System Stable"

    print("\n==============================")
    print(f"CPU: {cpu}% | RAM: {ram}% | Processes: {processes}")
    print(status)
    print(f"⏳ Estimated time to anomaly: {rul_minutes:.2f} minutes")
    print("==============================")

    time.sleep(5)
