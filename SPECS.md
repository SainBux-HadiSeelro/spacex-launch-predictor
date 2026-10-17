# 📋 Technical Specifications Document (SPECS)

## Project: SpaceX Launch Analysis & Prediction System

---

## 1. System Overview

The **SpaceX Launch Analysis & Prediction System** is a data science solution designed to process, visualize, and predict outcomes for SpaceX orbital launch missions. The system consists of:
- An Exploratory Data Analysis (EDA) and model training pipeline implemented in Jupyter Notebook (`smit-hackathon.ipynb`).
- A production inference artifact (`random_forest_model.pkl`).
- An interactive web application built with Streamlit (`app.py`).

---

## 2. Runtime & Dependency Specifications

### 2.1 Environment
- **Runtime**: Python 3.9, 3.10, 3.11, or 3.12 (64-bit)
- **Supported Operating Systems**: Windows 10/11, macOS, Linux (Ubuntu 20.04+)
- **Application Server**: Streamlit Local Web Server (Tornado-based, Default Port: `8501`)

### 2.2 Core Libraries & Minimum Versions
| Package | Minimum Version | Purpose |
|---|---|---|
| `streamlit` | `>=1.33.0` | Reactive web UI and dashboard controls |
| `pandas` | `>=1.5.3` | Tabular data manipulation and preprocessing |
| `numpy` | `>=1.24.0` | Numerical array processing and matrices |
| `scikit-learn` | `>=1.3.0` | Machine learning model loading and inference |
| `folium` | `>=0.14.0` | Geospatial mapping and Leaflet integration |
| `streamlit-folium`| `>=0.12.0` | Streamlit bi-directional wrapper for Folium |
| `joblib` | `>=1.2.0` | Model serialization and deserialization |
| `matplotlib` | `>=3.7.0` | Static data visualization (in EDA notebook) |
| `seaborn` | `>=0.12.0` | Statistical data plotting (in EDA notebook) |

---

## 3. Data Schema Specifications

### 3.1 `spacex_cleaned.csv` (Primary Production Dataset)
| Column Name | Data Type | Description | Sample Values |
|---|---|---|---|
| `mission_name` | String | Name of the SpaceX mission | `FalconSat`, `Starlink 4-21 (v1.5)` |
| `launch_date` | String (ISO-8601) | Launch timestamp in UTC | `2006-03-24T22:30:00.000Z` |
| `rocket_name` | String | Rocket family name | `Falcon 1`, `Falcon 9`, `Falcon Heavy` |
| `payload_mass` | Float | Payload mass in kilograms | `20.0`, `5000.0`, `15600.0` |
| `orbit` | String | Target orbital class | `LEO`, `VLEO`, `GTO`, `ISS`, `SSO`, `PO` |
| `site_name` | String | Launch complex site code | `CCSFS SLC 40`, `KSC LC 39A`, `VAFB SLC 4E`, `Kwajalein Atoll` |
| `location` | String | Geographical region name | `Cape Canaveral`, `Vandenberg Space Force Base`, `Omelek Island` |
| `success` | Boolean / Null | Binary launch outcome | `True`, `False`, `NaN` |

### 3.2 Supplemental Datasets
- `launches.csv`: Raw API launch records including fairing recovery, stage reusability, and crew details.
- `launchpads.csv`: Launchpad geodetic positions, regional boundaries, status, and launch attempt counters.
- `payloads.csv`: Individual payload manifest data (mass, customers, orbit regimes).
- `rockets.csv`: Rocket vehicle hardware specifications (stages, engines, height, diameter).

---

## 4. Geospatial Specifications

### 4.1 Coordinate Reference System (CRS)
- **CRS Standard**: WGS84 (`EPSG:4326`)
- **Map Projection**: Web Mercator (`EPSG:3857`)
- **Base Tile Provider**: CartoDB Positron / OpenStreetMap

### 4.2 Launchpad Geodetic Reference Table
| Launch Complex Site | Location Region | Latitude (° N) | Longitude (° E) |
|---|---|---|---|
| **CCSFS SLC 40** | Cape Canaveral, Florida | `28.5618571` | `-80.5773660` |
| **KSC LC 39A** | Merritt Island, Florida | `28.6080585` | `-80.6039558` |
| **VAFB SLC 4E** | Santa Barbara County, California | `34.6320930` | `-120.6108290` |
| **Kwajalein Atoll** | Omelek Island, Marshall Islands | `9.0477206` | `167.7431292` |

---

## 5. Machine Learning Specifications

### 5.1 Model Architecture
- **Model Type**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters**:
  - `n_estimators`: `100`
  - `criterion`: `'gini'`
  - `max_depth`: None (unconstrained until leaves are pure)
  - `min_samples_split`: `2`
  - `min_samples_leaf`: `1`
  - `random_state`: `42`
- **Output Target**: Binary Classification (`0` = Launch Failure / Anomaly, `1` = Successful Launch)

### 5.2 Feature Vector Schema (Inference)
The model expects an input matrix of **429 features**, structured as:
1. `payload_mass` (Continuous float, `0.0` to `16000.0` kg)
2. `year` (Integer, `2006` to `2026`)
3. One-hot encoded indicators:
   - `rocket_name_*`: `['rocket_name_Falcon 1', 'rocket_name_Falcon 9', 'rocket_name_Falcon Heavy']`
   - `site_name_*`: `['site_name_CCSFS SLC 40', 'site_name_KSC LC 39A', 'site_name_Kwajalein Atoll', 'site_name_VAFB SLC 4E']`
   - `location_*`: `['location_Cape Canaveral', 'location_Omelek Island', 'location_Vandenberg Space Force Base']`
   - `orbit_*`: `['orbit_ES-L1', 'orbit_GEO', 'orbit_GTO', 'orbit_HCO', 'orbit_HEO', 'orbit_ISS', 'orbit_LEO', 'orbit_MEO', 'orbit_PO', 'orbit_SO', 'orbit_SSO', 'orbit_TLI', 'orbit_VLEO']`
   - Historical mission name & timestamp dummy features (zero-filled during runtime prediction).

### 5.3 Inference Logic
- Missing inputs are zero-filled to maintain strict vector alignment with `model.feature_names_in_`.
- Probability scoring uses `model.predict_proba(input_data)[0]`.
- Decision threshold: `P(Success) >= 0.50` triggers a Success outcome.

---

## 6. Application Architecture & Optimization

### 6.1 Caching Strategy
- **`@st.cache_data`**: Applied to `load_data()` to prevent re-reading `spacex_cleaned.csv` on every user interaction.
- **`@st.cache_resource`**: Applied to `load_model()` to maintain a single in-memory instance of the deserialized Scikit-Learn Random Forest estimator.

### 6.2 UI & Layout Principles
- Responsive wide container layout (`layout="wide"`).
- Dynamic filtering using vectorized Pandas operations (`.copy()`, conditional Boolean masks).
- Fast Leaflet map rendering via `folium.plugins.MarkerCluster` to prevent browser DOM lag when rendering 200+ markers.
