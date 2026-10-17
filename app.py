import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib

# Set page layout and title
st.set_page_config(
    page_title="SpaceX Launch Analysis & Prediction",
    page_icon="🚀",
    layout="wide"
)

# Launch site coordinates mapping
LAUNCHPAD_COORDS = {
    'CCSFS SLC 40': (28.5618571, -80.577366),
    'KSC LC 39A': (28.6080585, -80.6039558),
    'VAFB SLC 4E': (34.632093, -120.610829),
    'Kwajalein Atoll': (9.0477206, 167.7431292)
}

# Site to location name mapping
SITE_TO_LOCATION = {
    'CCSFS SLC 40': 'Cape Canaveral',
    'KSC LC 39A': 'Cape Canaveral',
    'Kwajalein Atoll': 'Omelek Island',
    'VAFB SLC 4E': 'Vandenberg Space Force Base'
}

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("spacex_cleaned.csv")
    df['launch_date_dt'] = pd.to_datetime(df['launch_date'], errors='coerce')
    df['year'] = df['launch_date_dt'].dt.year.fillna(0).astype(int)
    # Normalize success column to boolean (True/False/None)
    df['is_success'] = df['success'].apply(
        lambda x: True if x is True or x == 1 else (False if x is False or x == 0 else None)
    )
    return df

# Load trained model with caching
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

# Initialize data and model
df = load_data()
model = load_model()
model_features = getattr(model, 'feature_names_in_', [])

st.title("🚀 SpaceX Launch Analysis & Prediction Dashboard")
st.markdown("Explore SpaceX launch history, geospatial launch sites, and predict launch success probability using Machine Learning.")

# Overview KPIs / Metrics
total_launches = len(df)
successful_launches = int((df['is_success'] == True).sum())
failed_launches = int((df['is_success'] == False).sum())
success_rate = (successful_launches / total_launches * 100) if total_launches > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Launches", total_launches)
col2.metric("Successful Launches", successful_launches)
col3.metric("Failed Launches", failed_launches)
col4.metric("Overall Success Rate", f"{success_rate:.1f}%")

st.divider()

# Historical Launch Data Viewer
st.header("📊 Launch History Viewer")

years = sorted([y for y in df['year'].unique() if y > 0])
launchpads = sorted(df['site_name'].dropna().unique())
rockets = sorted(df['rocket_name'].dropna().unique())

f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    selected_year = st.selectbox("Filter by Year", options=["All"] + years)
with f_col2:
    selected_pad = st.selectbox("Filter by Launchpad", options=["All"] + launchpads)
with f_col3:
    selected_rocket = st.selectbox("Filter by Rocket", options=["All"] + rockets)

filtered_df = df.copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df['year'] == selected_year]
if selected_pad != "All":
    filtered_df = filtered_df[filtered_df['site_name'] == selected_pad]
if selected_rocket != "All":
    filtered_df = filtered_df[filtered_df['rocket_name'] == selected_rocket]

st.markdown(f"**Showing {len(filtered_df)} launch record(s):**")
display_cols = ['mission_name', 'launch_date', 'rocket_name', 'payload_mass', 'orbit', 'site_name', 'location', 'is_success']
display_df = filtered_df[[c for c in display_cols if c in filtered_df.columns]].copy()
display_df.rename(columns={'is_success': 'success'}, inplace=True)
st.dataframe(display_df, use_container_width=True)

st.divider()

# Geospatial Launch Sites Map
st.header("🗺️ Launch Sites Map")
st.markdown("Geographic distribution of SpaceX launch sites. Green circles indicate successful missions, red indicate failures.")

# Base map centered between US and Pacific
launch_map = folium.Map(location=[28.5, -80.6], zoom_start=4, tiles="CartoDB positron")
marker_cluster = MarkerCluster().add_to(launch_map)

for _, row in filtered_df.iterrows():
    site = row.get('site_name')
    # Get exact coordinates for the launchpad
    if site in LAUNCHPAD_COORDS:
        lat, lon = LAUNCHPAD_COORDS[site]
    else:
        # Fallback to Cape Canaveral
        lat, lon = 28.5618571, -80.577366

    is_succ = row.get('is_success')
    color = "green" if is_succ is True else ("red" if is_succ is False else "gray")
    status_label = "Success" if is_succ is True else ("Failure" if is_succ is False else "Unknown")
    mission = row.get('mission_name', 'Unknown')
    rocket = row.get('rocket_name', 'Unknown')
    orbit_val = row.get('orbit', 'N/A')

    folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(f"<b>Mission:</b> {mission}<br><b>Rocket:</b> {rocket}<br><b>Site:</b> {site}<br><b>Orbit:</b> {orbit_val}<br><b>Status:</b> {status_label}", max_width=300)
    ).add_to(marker_cluster)

st_folium(launch_map, width="100%", height=500)

st.divider()

# Prediction Tool
st.header("🔮 Launch Success Predictor")
st.markdown("Enter launch parameters to estimate the probability of mission success using the trained Random Forest model:")

p_col1, p_col2 = st.columns(2)
with p_col1:
    payload_mass = st.slider("Payload Mass (kg)", min_value=0, max_value=16000, value=5000, step=100)
    year = st.slider("Launch Year", min_value=2006, max_value=2026, value=2024)
    rocket_name = st.selectbox("Rocket Name", sorted(df['rocket_name'].dropna().unique()))

with p_col2:
    launchpad_name = st.selectbox("Launchpad Name", sorted(df['site_name'].dropna().unique()))
    orbits = sorted([o for o in df['orbit'].dropna().unique()])
    orbit = st.selectbox("Target Orbit", options=orbits)

# Build feature dictionary aligned with training features
location_name = SITE_TO_LOCATION.get(launchpad_name, '')
input_dict = {
    'payload_mass': payload_mass,
    'year': year,
    f'rocket_name_{rocket_name}': 1,
    f'site_name_{launchpad_name}': 1,
    f'location_{location_name}': 1,
    f'orbit_{orbit}': 1
}

# Construct dataframe matching model feature schema
input_data = pd.DataFrame(0, index=[0], columns=model_features)

for col, val in input_dict.items():
    if col in input_data.columns:
        input_data.at[0, col] = val

input_data.fillna(0, inplace=True)

if st.button("Predict Launch Success", type="primary"):
    try:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        success_prob = prob[1] if len(prob) > 1 else (1.0 if pred == 1 else 0.0)

        st.subheader("Prediction Result:")
        if pred == 1:
            st.success(f"✅ **Predicted Outcome: SUCCESS** (Confidence: {success_prob * 100:.1f}%)")
        else:
            fail_prob = prob[0] if len(prob) > 1 else (1.0 - success_prob)
            st.error(f"⚠️ **Predicted Outcome: FAILURE / ANOMALY** (Failure Risk: {fail_prob * 100:.1f}%)")

        st.progress(float(success_prob))
        st.caption(f"Estimated Success Probability: {success_prob:.2%}")

    except Exception as e:
        st.error("Prediction failed. Error details:")
        st.error(str(e))
