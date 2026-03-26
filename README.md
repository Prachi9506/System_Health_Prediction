#System Health Prediction

This project is an system health monitoring tool that collects real-time PC performance data, detects anomalies using machine learning, and predicts the Remaining Useful Life (RUL) of the system performance. It features a live Streamlit dashboard for visual monitoring and multiple scripts for data collection, model training, and real-time inference.

## Project Structure

* **`pc_data_collector.py`**: Gathers system metrics (CPU, RAM, Disk, Network, etc.) using `psutil` and saves them to `pc_health_data.csv`.
* **`train_model.py` / `anomoly.py`**: Trains an Isolation Forest model to identify unusual system behavior and saves it as `anomaly_model.pkl`.
* **`generate_rul_data.py`**: Processes collected health data to create a Remaining Useful Life (RUL) dataset.
* **`train_rul_model.py`**: Trains a Random Forest Regressor to predict the time remaining before a potential anomaly occurs.
* **`dashboard.py`**: A Streamlit-based web dashboard that visualizes live system stats, anomaly status, and RUL predictions.
* **`realtime_monitor.py` / `realtime_rul_monitor.py`**: Console-based scripts for continuous anomaly and RUL monitoring without the dashboard.
* **`anomaly_score_plot.py`**: Generates visualizations of anomaly scores over time for analysis.

## Features

* **Real-time Monitoring**: Tracks CPU usage, virtual memory, disk usage, and network activity.
* **Anomaly Detection**: Uses an Isolation Forest algorithm to detect deviations from normal system performance.
* **Predictive Maintenance**: Estimates "Time to Anomaly" using a trained regression model.
* **Visual Dashboard**: Live line charts for CPU/RAM and status indicators (Stable, Moderate Risk, Critical).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Prachi9506/System_Health_Prediction
    cd system_health_prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas sklearn psutil streamlit joblib matplotlib seaborn
    ```
