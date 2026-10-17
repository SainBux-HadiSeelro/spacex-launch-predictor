# 🚀 SpaceX Launch Analysis & Machine Learning Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/pandas-1.5%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Folium](https://img.shields.io/badge/folium-0.14%2B-77B800.svg?logo=leaflet&logoColor=white)](https://python-visualization.github.io/folium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Data Science and Machine Learning web application that analyzes historical SpaceX launches, visualizes launchpad sites on an interactive world map, and predicts launch success probability using a trained Random Forest model.

---

## 📌 Project Overview

This project provides data-driven insights into SpaceX orbital launches by leveraging raw telemetry, launch records, payloads, and launchpad details. It bridges exploratory data analysis (EDA) with production-ready interactive dashboards and predictive modeling.

### ✨ Key Features

- **📊 Comprehensive Historical Analytics**:
  - Filter and inspect 200+ historical SpaceX missions by launch year, launchpad site, and rocket model (Falcon 1, Falcon 9, Falcon Heavy).
  - Dynamic KPI cards computing total launches, successful vs. failed missions, and live success rate percentages.

- **🗺️ Interactive Geospatial Mapping**:
  - Global map visualization powered by Folium and Leaflet.js.
  - Accurate GPS coordinates for all major SpaceX launch complexes (Cape Canaveral CCSFS SLC-40, Kennedy Space Center KSC LC-39A, Vandenberg VAFB SLC-4E, and Kwajalein Atoll).
  - Clustered mission markers with color-coded success/failure indicators and rich popup telemetry.

- **🔮 Machine Learning Launch Predictor**:
  - Real-time mission outcome prediction powered by a pre-trained **Random Forest Classifier**.
  - Interactive sliders and dropdowns for payload mass (0–16,000 kg), launch year, target orbit (VLEO, LEO, GTO, ISS, etc.), rocket type, and launch site.
  - Outputs predicted outcome (Success vs. Failure/Anomaly) alongside confidence probabilities and dynamic progress indicators.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend / Dashboard** | [Streamlit](https://streamlit.io/), [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium) |
| **Data Processing & EDA** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Geospatial Visuals** | [Folium](https://python-visualization.github.io/folium/), [Leaflet.js](https://leafletjs.com/) |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/) |
| **Data Visualization (EDA)** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application
├── smit-hackathon.ipynb        # Jupyter Notebook for EDA, data cleaning & model training
├── random_forest_model.pkl     # Pre-trained Random Forest Classifier
├── requirements.txt            # Python dependencies
├── SPECS.md                    # Detailed Technical & System Specifications
├── spacex_cleaned.csv          # Cleaned dataset used by the dashboard
├── launches.csv                # SpaceX raw launches dataset
├── launchpads.csv              # Launch site geographical & metadata records
├── payloads.csv                # Payload specifications dataset
└── rockets.csv                 # Rocket specifications dataset
```

---

## 🔬 Machine Learning Pipeline

1. **Data Ingestion & Cleaning**:
   - Merged telemetry across launches, payloads, rockets, and launchpads.
   - Handled missing payload mass values and target status encodings.
2. **Feature Engineering**:
   - Extraction of launch calendar year from ISO-8601 timestamps.
   - Categorical feature encoding (One-Hot Encoding for rocket name, orbit class, launchpad, and location).
3. **Model Selection & Training**:
   - **Algorithm**: Random Forest Classifier (`n_estimators=100`, `random_state=42`).
   - **Input Dimensions**: Aligned feature matrix mapping payload mass, temporal factors, and encoded categorical attributes.
   - Evaluated using train/test splits with precision, recall, and ROC-AUC metrics.

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SainBux-HadiSeelro/smit-hackathon.git
cd smit-hackathon
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🎯 Usage Walkthrough

1. **Dashboard Overview**: Review overall SpaceX mission statistics at the top of the page.
2. **Filter Launches**: Use the dropdown filters to narrow down launches by specific launchpad, rocket type, or year.
3. **Explore Launch Sites**: Zoom into the interactive map to see launch clusters around Florida, California, and the Marshall Islands. Click any pin to view mission details.
4. **Predict Mission Outcome**:
   - Set the planned payload mass in kg.
   - Choose the target orbit and launch year.
   - Select the rocket vehicle and launch site.
   - Click **Predict Launch Success** to see the machine learning model's probability estimate.

---

## 👤 Author

**Sain Bux (Hadi Seelro)**
- GitHub: [@SainBux-HadiSeelro](https://github.com/SainBux-HadiSeelro)
- Repository: [smit-hackathon](https://github.com/SainBux-HadiSeelro/smit-hackathon)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
