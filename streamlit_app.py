import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from streamlit_folium import st_folium
from datetime import datetime
import seaborn as sns
import folium
from folium.plugins import MarkerCluster

BASE_DIR = "data"

def load_json_logs(log_dir, input_dir="input", output_dir="output"):
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
def add_file_sizes(df, country, base_dir="data", input_dir="input", output_dir="output"):

    def get_size_mb(path):
        return round(os.path.getsize(path) / (1024 * 1024), 5) if os.path.exists(path) else None

    input_sizes = []
    output_sizes = []

    for idx, row in df.iterrows():
        image_id = row.get("image_id") or row.get("image_id_from_log")
        if not image_id:
            input_sizes.append(None)
            output_sizes.append(None)
            continue

        input_path = os.path.join(base_dir, country, input_dir, f"{image_id}")
        output_path = os.path.join(base_dir, country, output_dir, f"{image_id}")

        # print(f"Checking sizes for {input_path} and {output_path}")

        input_sizes.append(get_size_mb(input_path))
        output_sizes.append(get_size_mb(output_path))

    df["input_file_size"] = input_sizes
    df["output_file_size"] = output_sizes
    return df


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
            df = add_file_sizes(df, country=country)
            # print(df)
            country_dfs[country] = df
        except Exception as e:
            msg = f"Error processing logs for {country}: {e}"
            print(msg)
            st.warning(msg)
            continue

    return country_dfs

def plot_inference_time_and_file_sizes(df, country_name):
    df_sorted = df.sort_values("timestamp")

    chart = (alt.Chart(df_sorted)
   .mark_circle()
   .encode(x="image_id", y="inference_time_seconds", size="input_file_size", color="input_file_size", 
   tooltip=["image_id", "inference_time_seconds", "output_file_size", "input_file_size"])
   )

    chart = chart.properties(
        title=f"Inference Time and File Sizes for {country_name}",
        width=800,
        height=400
    )

    return chart

def plot_file_sizes(df, country_name):
    # Check required columns
    expected_cols = ['image_id', 'input_file_size', 'output_file_size']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns in data: {missing_cols}")
        return None

    df_sorted = df.sort_values("timestamp")
    # Melt for grouped bar chart
    melted_df = df_sorted[['image_id', 'input_file_size', 'output_file_size']].melt(
        id_vars='image_id',
        value_vars=['input_file_size', 'output_file_size'],
        var_name='File Type',
        value_name='Size (MB)'
    )

    # Make sure File Type is categorical and ordered
    melted_df['File Type'] = melted_df['File Type'].map({
        'input_file_size': 'Input',
        'output_file_size': 'Output'
    })


    # Instead of using column, use x with offset for grouped bars
    chart = alt.Chart(melted_df).mark_bar().encode(
        x=alt.X('image_id:N', title='Image ID', axis=alt.Axis(
            labelExpr="substring(datum.value, 0, 8)",
            labelAngle=-45
        )),
        y=alt.Y('Size (MB):Q', title='File Size (MB)'),
        color=alt.Color('File Type:N', scale=alt.Scale(domain=['Input', 'Output'], range=['#1f77b4', '#ff7f0e'])),
        xOffset='File Type:N',
        tooltip=['image_id', 'File Type', 'Size (MB)']
    ).properties(
        title=f'File Sizes for {country_name}',
        width=20 * max(1, len(melted_df["image_id"].unique())),
        height=400
    )

    return chart


def plot_inference_time_altair(df, country):
    df_sorted = df.sort_values("timestamp")

    # c = (alt.Chart(df).mark_circle().encode(x="image_id", y="inference_time_seconds"))
    #c = c.properties(title=f"Image Inference Time per Image – {country}")

    chart = alt.Chart(df_sorted).mark_bar(color='teal').encode(
        x=alt.X('image_id', sort=None, title='Image ID'),
        y=alt.Y('inference_time_seconds', title='Inference Time (s)'),
        tooltip=['image_id', 'inference_time_seconds']
    ).properties(
        title=f'Image Inference Time per Image – {country}',
        width=800,
        height=400
    )

    return chart

def show_map(df):
    m = folium.Map(location=[0, 0], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        loc = row.get("location", {})
        lat, lon = loc.get("lat"), loc.get("lon")
        if lat is not None and lon is not None:
            tooltip = f"image id: {row['image_id'][:6]} inference time: {row['inference_time_seconds']}s"
            folium.Marker([lat, lon], tooltip=tooltip).add_to(marker_cluster)

    return m    


def show_dashboard():
    st.set_page_config(page_title="Satlyt Testbed Monitoring Dashboard", layout="wide")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("satlyt_logo_light.png", width=300)  # Replace with your logo URL
        st.markdown("---", width=300)

    st.title("Testbed Inference Dashboard")

    all_dfs = load_country_logs()

    if not all_dfs:
        st.warning("No logs found.")
        return

    country_list = [country.capitalize() for country in list(all_dfs.keys())]
    selected_country = st.sidebar.selectbox("🌍 Select Country", country_list)

    df = all_dfs[selected_country.lower()]
    if df.empty:
        st.warning(f"No data available for {selected_country}.")
        return

    st.subheader("📉 Inference Time")
    chart = plot_inference_time_altair(df, selected_country)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("---")


    st.subheader("📂 Input vs Output File Size")
    file_size_chart = plot_file_sizes(df, selected_country)
    if file_size_chart:
        st.altair_chart(file_size_chart, use_container_width=True)

    st.markdown("---")


    st.subheader("📊 Inference Time and File Sizes")
    combined_chart = plot_inference_time_and_file_sizes(df, selected_country)
    if combined_chart:
        st.altair_chart(combined_chart, use_container_width=True)
    st.markdown("---")

    st.subheader("🗺️ Location Map")
    fmap = show_map(df)
    if fmap:
        st.write("Map of Image Locations")
        st_folium(fmap, use_container_width=True)

        st.write("\n\n")
        st.write("Hover over markers to see image ID, timestamp, and inference time.")
        st.write("Markers represent images with their inference times.")
        st.write("\n\nMap shows locations of images with inference times.")
            
    else:
        st.info("Map not available.")

    


if __name__ == "__main__":
    show_dashboard()
