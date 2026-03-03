import psutil
import pandas as pd
import time
from datetime import datetime
import os

file_name = "pc_health_data.csv"

if not os.path.exists(file_name):
    df = pd.DataFrame(columns=[
        "timestamp",
        "cpu_usage",
        "ram_usage",
        "disk_usage",
        "net_sent",
        "net_recv",
        "process_count",
        "uptime_seconds",
        "label"
    ])
    df.to_csv(file_name, index=False)

print("Starting PC data collection... Press CTRL+C to stop.")

boot_time = psutil.boot_time()

while True:
    timestamp = datetime.now()

    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent

    net_io = psutil.net_io_counters()
    net_sent = net_io.bytes_sent
    net_recv = net_io.bytes_recv

    processes = len(psutil.pids())

    uptime = time.time() - boot_time

    health_score = (cpu + ram + disk) / 3

    if health_score > 70:
        label = 1   # Unhealthy
    else:
        label = 0   # Healthy

    row = pd.DataFrame([[
        timestamp, cpu, ram, disk,
        net_sent, net_recv,
        processes, uptime,
        label
    ]], columns=[
        "timestamp",
        "cpu_usage",
        "ram_usage",
        "disk_usage",
        "net_sent",
        "net_recv",
        "process_count",
        "uptime_seconds",
        "label"
    ])

    row.to_csv(file_name, mode='a', header=False, index=False)

    print(f"Logged: CPU={cpu}% RAM={ram}% Label={label}")

    time.sleep(4)
