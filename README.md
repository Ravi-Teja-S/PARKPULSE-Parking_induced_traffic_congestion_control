# PARKPULSE: Parking-Induced Traffic Congestion Control

A Streamlit-based dashboard for detecting traffic congestion hotspots, forecasting high-risk zones, and supporting proactive traffic management.

---

## Problem Statement

Traffic congestion caused by parking violations and vehicle incidents leads to increased travel time, reduced road efficiency, and delayed emergency response.

PARKPULSE uses machine learning and geospatial analytics to detect congestion hotspots in real time, predict future congestion risk, and assist traffic authorities in making informed operational decisions.

---

## Features

- Real-time congestion hotspot detection
- Next-hour congestion risk prediction using XGBoost
- Interactive maps and visualizations
- Dispatch queue simulation
- What-if scenario analysis
- SHAP-based model explainability

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- XGBoost
- HDBSCAN
- SHAP
- Folium
- PyDeck
- Scikit-learn

---

## Team AI-One

- Ravi Teja S
- Pratham Manoj Patil
- Praveen Kumar Y

---

## Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ravi-Teja-S/PARKPULSE-Parking_induced_traffic_congestion_control.git
cd PARKPULSE-Parking_induced_traffic_congestion_control
```

### 2. Create and activate a virtual environment

```bash
uv venv
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## Project Structure

```text
PARKPULSE/
├── app.py
├── requirements.txt
├── models/
    └──forecast_model.pkl
├── data/
    └──processed
    └──raw(dataset given by the hackathon)
├── notebooks/
├── .streamlit/
     └──secrets.toml
└── README.md
```

---

## Notes

- The repository includes the trained model and processed datasets.
- A valid Gemini API key is required for AI-powered features.
- Built as a hackathon project for intelligent traffic congestion management.

## Future enhancements

- Add live GPS/travel-time feeds for more accurate real-time congestion detection.
- Integrate multi-modal alerts to dispatch traffic control units faster.
- Expand the model to include weather and event schedule data for better forecast accuracy.
- Add a mobile-friendly incident response view for field teams.
