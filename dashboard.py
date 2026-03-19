import streamlit as st
import pandas as pd

st.title("System Health Monitor")

df = pd.read_csv("anomaly_log.csv")

st.write("Recent Anomalies:")
st.dataframe(df.tail(10))

st.line_chart(df["cpu_usage"])
st.line_chart(df["ram_usage"])
