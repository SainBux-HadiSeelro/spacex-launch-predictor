# 🚀 SpaceX Launch Analysis & Machine Learning Prediction Dashboard

[![Live Demo](https://img.shields.io/badge/Live%20Demo-spacex--launch--predictor--hadi.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://spacex-launch-predictor-hadi.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SainBux-HadiSeelro/smit-hackathon)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Folium](https://img.shields.io/badge/folium-0.14%2B-77B800.svg?logo=leaflet&logoColor=white)](https://python-visualization.github.io/folium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🔴 **Live Web Application:** [https://spacex-launch-predictor-hadi.streamlit.app/](https://spacex-launch-predictor-hadi.streamlit.app/)  
> 💻 **Source Code:** [https://github.com/SainBux-HadiSeelro/smit-hackathon](https://github.com/SainBux-HadiSeelro/smit-hackathon)

---

## 📖 About The Project

Space exploration and orbital rocket launches involve astronomical financial investments, complex engineering tolerances, and extreme environmental variables. With each Falcon 9 launch costing upwards of **$60+ Million**, understanding the factors that dictate launch outcomes—and accurately predicting mission success before ignition—is paramount for commercial spaceflight operators and satellite customers.

This project delivers a **full-lifecycle Data Science and Machine Learning solution** that:
1. **Aggregates and Cleans SpaceX Telemetry**: Ingests multi-source historical datasets covering 200+ launches, rocket specifications, payload characteristics, and orbital regimes.
2. **Performs In-Depth Exploratory Data Analysis (EDA)**: Dissects payload distributions, launch site efficiencies, and year-over-year success trajectory improvements.
3. **Deploys Geospatial Intelligence**: Maps global launchpad complexes with true geodetic coordinates (Cape Canaveral, Vandenberg Space Force Base, and Kwajalein Atoll) using interactive clustering and telemetry popups.
4. **Predicts Mission Outcomes in Real-Time**: Integrates a trained **Random Forest Classifier** that takes user-defined payload mass, orbital destination, vehicle class, and launchpad location to estimate the probability of success with dynamic confidence scoring.

---

## ✨ Key Features

- **📊 Comprehensive Historical Analytics**:
  - Filter 200+ historical SpaceX missions by calendar year, launch site, and rocket vehicle (Falcon 1, Falcon 9, Falcon Heavy).
  - Dynamic KPI cards computing total launches, successful vs. failed missions, and real-time success rate percentages.

- **🗺️ Interactive Geospatial Mapping**:
  - Global interactive Leaflet map powered by Folium and Streamlit.
  - Pinpoints exact GPS coordinates for all SpaceX launch sites:
    - **CCSFS SLC-40** & **KSC LC-39A** (Cape Canaveral, FL)
    - **VAFB SLC-4E** (Vandenberg SFB, CA)
    - **Kwajalein Atoll** (Omelek Island, Marshall Islands)
  - Clustered mission markers with color-coded success/failure indicators and rich popup telemetry.

- **🔮 Machine Learning Launch Predictor**:
  - Real-time mission outcome prediction powered by a pre-trained **Random Forest Classifier**.
  - Interactive sliders and dropdowns for payload mass (0–16,000 kg), launch year, target orbit (VLEO, LEO, GTO, ISS, etc.), rocket type, and launch site.
  - Outputs predicted outcome (Success vs. Failure/Anomaly) alongside confidence probabilities and dynamic progress indicators.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend / Web App** | [Streamlit](https://streamlit.io/), [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium) |
| **Data Processing & EDA** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Geospatial Visuals** | [Folium](https://python-visualization.github.io/folium/), [Leaflet.js](https://leafletjs.com/) |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/) |
| **Data Visualization (EDA)** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Cloud Hosting** | [Streamlit Community Cloud](https://share.streamlit.io/) |

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application
├── smit-hackathon.ipynb        # Jupyter Notebook for EDA, data cleaning & model training
├── random_forest_model.pkl     # Pre-trained Random Forest Classifier
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & portfolio showcase
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

## ⚡ Quick Start & Local Installation

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

## 🌐 Live Application Link

You can access and interact with the deployed application directly in your browser without any setup:

👉 **[Launch SpaceX Predictor Dashboard](https://spacex-launch-predictor-hadi.streamlit.app/)**

---

## 👤 Author

**Sain Bux (Hadi Seelro)**
- **Live App**: [spacex-launch-predictor-hadi.streamlit.app](https://spacex-launch-predictor-hadi.streamlit.app/)
- **GitHub**: [@SainBux-HadiSeelro](https://github.com/SainBux-HadiSeelro)
- **Repository**: [smit-hackathon](https://github.com/SainBux-HadiSeelro/smit-hackathon)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
