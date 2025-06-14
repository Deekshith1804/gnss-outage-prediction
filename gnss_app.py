import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# ---------------------- Title ---------------------- #
st.title("📡 GNSS Outage Prediction Based on Weather & Ionospheric Data")

# ------------------ Sidebar Controls ------------------ #
st.sidebar.header("⚙️ Simulation Controls")
n_rows = st.sidebar.slider("Number of Data Points", min_value=100, max_value=2000, value=1000, step=100)

# -------------------- Simulate Data ------------------- #
dates = pd.date_range(start="2025-06-09", periods=n_rows, freq="H")
data = pd.DataFrame({
    "timestamp": dates,
    "latitude": np.random.uniform(10, 30, n_rows),
    "longitude": np.random.uniform(70, 90, n_rows),
    "rain": np.random.uniform(0, 50, n_rows),
    "cloud_cover": np.random.uniform(0, 100, n_rows),
    "TEC": np.random.uniform(1, 100, n_rows),
    "geomagnetic": np.random.uniform(10, 500, n_rows),
})

# ------------- Label: Outage Based on Rules ------------- #
data["outage"] = (
    (data["TEC"] > 70) &
    (data["geomagnetic"] > 300) &
    (data["cloud_cover"] > 70)
).astype(int)

# -------------------- Show Raw Data -------------------- #
if st.checkbox("🔍 Show Raw Data"):
    st.write(data.head())

# --------------------- Train Model --------------------- #
features = ["rain", "cloud_cover", "TEC", "geomagnetic"]
X_train, X_test, y_train, y_test = train_test_split(data[features], data["outage"], test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- Model Performance Report ---------------- #
st.subheader("📊 Model Performance")
st.text(classification_report(y_test, y_pred))

# ------------------- Create HeatMap ------------------- #
st.subheader("🗺️ GNSS Outage Locations")
heat_data = data[data["outage"] == 1][["latitude", "longitude"]].values.tolist()
m = folium.Map(location=[20, 80], zoom_start=5)
HeatMap(heat_data).add_to(m)

# ---------- Custom Prediction from Location Name ---------- #
st.sidebar.header("📍 Custom Outage Prediction")

location_name = st.sidebar.text_input("Enter Location Name (e.g., Hyderabad, India)", "Hyderabad, India")
geolocator = Nominatim(user_agent="gnss_app")
location = geolocator.geocode(location_name)

if location:
    custom_lat = location.latitude
    custom_lon = location.longitude
    st.sidebar.success(f"📌 Located: {location.address}")
else:
    st.sidebar.error("❌ Location not found. Using default coordinates.")
    custom_lat = 17.4
    custom_lon = 78.5

# ---------- Fixed Input Values for Prediction ---------- #
custom_input = pd.DataFrame([{
    "rain": 10.0,
    "cloud_cover": 50.0,
    "TEC": 30.0,
    "geomagnetic": 200.0
}])

custom_pred = model.predict(custom_input)[0]
custom_proba = model.predict_proba(custom_input)[0][1]

# ---------- Display Prediction Result ---------- #
st.sidebar.markdown("### 🔎 Prediction:")
st.sidebar.info("Using default atmospheric and ionospheric values.")
st.sidebar.success("🔴 GNSS Outage Likely" if custom_pred else "🟢 No Outage Expected")
st.sidebar.progress(custom_proba)

# ---------- Marker on Map ---------- #
folium.Marker(
    [custom_lat, custom_lon],
    popup=f"{location_name}<br>Outage: {'Yes' if custom_pred else 'No'}<br>Prob: {custom_proba:.2f}",
    icon=folium.Icon(color="red" if custom_pred else "green")
).add_to(m)

# ---------- Show Map ---------- #
folium_static(m)

# ---------- Download CSV ---------- #
st.download_button(
    label="⬇️ Download Dataset as CSV",
    data=data.to_csv(index=False),
    file_name="gnss_outage_dataset.csv",
    mime="text/csv"
)
