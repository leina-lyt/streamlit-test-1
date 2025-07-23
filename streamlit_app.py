import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import folium_static
from datetime import datetime
import seaborn as sns

BASE_DIR = "data"

def load_json_logs(log_dir):
    logs = []
    if not os.path.exists(log_dir):
        msg = f"Directory not found: {log_dir}"
        print(msg)
        st.warning(msg)
        return pd.DataFrame()

    json_files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
    if not json_files:
        msg = f"No JSON files found in {log_dir}"
        print(msg)
        st.warning(msg)
        return pd.DataFrame()

    for filename in json_files:
        full_path = os.path.join(log_dir, filename)
        with open(full_path, "r") as f:
            try:
                logs.append(json.load(f))
            except json.JSONDecodeError:
                msg = f"Warning: Could not decode {filename}"
                print(msg)
                st.warning(msg)

    return pd.DataFrame(logs)

def load_country_logs(base_dir=BASE_DIR):
    country_dfs = {}

    if not os.path.exists(base_dir):
        msg = f"Base directory not found: {base_dir}"
        print(msg)
        st.warning(msg)
        return {}

    for country in os.listdir(base_dir):
        input_dir = os.path.join(base_dir, country, "input_logs")
        output_dir = os.path.join(base_dir, country, "output_logs")

        if not (os.path.isdir(input_dir) and os.path.isdir(output_dir)):
            msg = f"Missing input or output folder for {country}"
            print(msg)
            st.warning(msg)
            continue

        input_df = load_json_logs(input_dir)
        output_df = load_json_logs(output_dir)

        if input_df.empty or output_df.empty:
            st.info(f"Skipping {country}: empty input or output logs.")
            continue

        try:
            df = pd.merge(input_df, output_df,
                          left_on="image_id", right_on="image_id_from_log",
                          how="inner")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["country"] = country
            country_dfs[country] = df
        except Exception as e:
            msg = f"Error processing logs for {country}: {e}"
            print(msg)
            st.warning(msg)

    return country_dfs

def plot_inference_time(df, country):
    df_sorted = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_sorted["image_id"], df_sorted["inference_time_seconds"], color="teal")
    ax.set_xlabel("Image ID (truncated)")
    ax.set_ylabel("Inference Time (s)")
    ax.set_title(f"Inference Time per Image – {country}")
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    return fig

def show_map(df):
    try:
        import folium
        from folium.plugins import MarkerCluster

        m = folium.Map(location=[0, 0], zoom_start=2)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df.iterrows():
            loc = row.get("location", {})
            lat, lon = loc.get("lat"), loc.get("lon")
            if lat is not None and lon is not None:
                tooltip = f"{row['image_id'][:6]} @ {row['timestamp']} – {row['inference_time_seconds']}s"
                folium.Marker([lat, lon], tooltip=tooltip).add_to(marker_cluster)

        return m
    except ImportError:
        st.error("Folium not installed")
        return None

def show_dashboard():
    st.title("📊 Inference Monitoring Dashboard")
    all_dfs = load_country_logs()

    if not all_dfs:
        st.warning("No logs found.")
        return

    for country, df in all_dfs.items():
        st.header(f"🌍 Country: {country}")
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("📉 Inference Time")
            fig = plot_inference_time(df, country)
            st.pyplot(fig)

        with col2:
            st.subheader("🗺️ Location Map")
            fmap = show_map(df)
            if fmap:
                folium_static(fmap)
            else:
                st.info("Map not available")

if __name__ == "__main__":
    show_dashboard()
