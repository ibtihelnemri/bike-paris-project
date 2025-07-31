import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os, sys
from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO
import json
import requests

# -------------------------------
# Data and model loading
# -------------------------------

st.set_page_config(page_title="Bike Traffic Prediction", layout="wide")
print(sys.path)


@st.cache_resource
def load_data():
    mode = os.getenv("MODE", "cloud")
    if mode == "local":
        df= pd.read_parquet("data/comptage_clean.parquet")
        df_counters = pd.read_csv("data/liste_compteurs.csv")
        model_reg = joblib.load("model.joblib")
        encoder_reg = joblib.load("encoder.joblib")
        model_clf = joblib.load("model_classifier.joblib")
        encoder_clf = joblib.load("encoder_classifier.joblib")
    else:
        credentials_info = json.loads(st.secrets["GCP_KEY"])
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        client = storage.Client(credentials=credentials, project=credentials.project_id)

        bucket = client.bucket("bike-data-bucket-ibtihel-2025")
        print("loading data")
        df = pd.read_parquet(BytesIO(bucket.blob("comptage_clean.parquet").download_as_bytes()))
        df_counters = pd.read_csv(BytesIO(bucket.blob("liste_compteurs.csv").download_as_bytes()))
        print("loading data is done")
        
    return df
    #return df

df = load_data()
#df = load_all_data_and_models()
df['hour'] = df['date_et_heure_de_comptage'].dt.hour
df['weekday'] = df['date_et_heure_de_comptage'].dt.day_name()
df['month'] = df['date_et_heure_de_comptage'].dt.month
df[['latitude', 'longitude']] = df['coordonnées_géographiques'].str.split(',', expand=True).astype(float)
df = df.dropna(subset=['latitude', 'longitude'])

# -------------------------------
# UI Navigation
# -------------------------------

st.sidebar.title("🤍 Navigation")
section = st.sidebar.radio("Choose a section", [
    "Project Overview", "Exploratory Analysis", "Approach", "Modeling", "Demo"])

image = Image.open("images/velo.jpg")
resized_image = image.resize((500, 200))
st.image(resized_image, use_container_width=True)

st.markdown("""
<h1 style='text-align: center; font-size: 2em; margin-top: -10px;'>
    Hourly Bike Traffic Prediction
</h1>
<p style='text-align: center; font-size: 1.6em; color: grey; margin-top: -10px;'>
    From November 2024 to November 2025
</p>
""", unsafe_allow_html=True)

# -------------------------------
# 1. Project Overview
# -------------------------------
if section == "Project Overview":
    st.markdown("""
<div style="border:1px solid #ccc; padding:20px; border-radius:10px; text-align:center; font-size:16px">

<p>
Project developed as part of the <strong>Machine Learning Engineer</strong> training by 
<a href="https://www.datascientest.com" target="_blank">DataScientest</a><br>
Cohort: November 2025
</p>

<p><strong>Authors:</strong><br>
<a href="https://www.linkedin.com/in/arthurcornelio/" target="_blank"><strong>Arthur CORNELIO</strong></a><br>
<a href="https://www.linkedin.com/in/brunohappi17/" target="_blank"><strong>Bruno HAPPI</strong></a><br>
<a href="https://www.linkedin.com/in/ibnemri/" target="_blank"><strong>Ibtihel NEMRI</strong></a>
</p>

<p><strong>Data source:</strong><br>
<a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs" target="_blank">
Bike Count | Open Data | City of Paris</a>
</p>

</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
---

## Project Objectives

This project aims to:

- Analyze Parisian bike traffic using data from automated counting devices  
- Identify high-traffic zones and peak hours  
- Build predictive models to forecast the number of cyclists based on location, date, and time  
- Provide useful indicators to help the City of Paris optimize cycling infrastructure  

---

## Methodology Summary

- Cleaning and structuring data: standardization, time feature extraction, geographic processing  
- Statistical analysis: peaks identification, hourly/daily/monthly averages  
- Visualizations: heatmaps, time-series graphs, peak traffic summaries  
- Development of a **Random Forest Regressor** to predict hourly bike traffic  
- Construction of a **Random Forest Classifier** to detect high traffic (crowding) situations  
- Deployment in an interactive Streamlit application  

---

## Examples of Visualizations

Below are some examples from the exploratory analysis:
""")

    st.image("images/heatmap_static.png", caption="Map of high bike traffic zones", use_container_width=True)
    st.image("images/traffic_hour_day.png", caption="Average hourly traffic by weekday", use_container_width=True)
    st.image("images/pics_barplot.png", caption="Top 20 peak traffic hours by counting site", use_container_width=True)

    st.markdown("""
---

## Deployed Application

The final models were integrated into a Streamlit app that allows users to:

- Make predictions for a given location, day, hour, and month  
- Explore both regression (hourly count) and classification (crowding) results  
- Get clear, interpretable feedback to support urban planning  

→ Use the **Demo** tab in the sidebar to interact with the interface.

---
""")


# -------------------------------
# 2. Exploratory Analysis
# -------------------------------
elif section == "Exploratory Analysis":
    st.title("Exploratory Analysis")
    with st.expander("About the Dataset", expanded=True):
        st.markdown("""
        Data collected by the City of Paris from ~100 automated counters includes:
        - Timestamp (hourly granularity)
        - Counter name and location
        - GPS coordinates
        - Bike counts per hour
        """)
        
    st.markdown("""
In this section, we explore hourly bike count data collected by the City of Paris.  
The goal is to understand **key cycling mobility trends**, highlight **behavior by hour, day, or location**, and identify **high-traffic areas**.

The analyses are based on raw data that we **cleaned, enriched, and structured**, making it suitable for statistical analysis and modeling.

### Data Loading and Preview

The dataset comes from the OpenData platform of the City of Paris.  
It records **hourly bike counts** by around one hundred sensors deployed across the city.  
Each row corresponds to **one hourly measurement at a specific counter**.

It contains:
- Date and time of measurement
- Name and location of the counter
- Number of bikes counted per hour
- GPS coordinates of the counting site
""")
    st.write(df.head())

    st.subheader("Data Cleaning")
    st.markdown("""
Before any analysis, several preprocessing steps were applied:

- Standardizing column names for easier manipulation
- Converting timestamps to `datetime` format with Paris time zone
- Creating useful temporal variables: **hour, day, month, weekday**
- Splitting **geographic coordinates** for map visualization
- Removing superfluous columns (image links, technical IDs, etc.)
- Optimizing categorical columns for memory efficiency
""")

    st.subheader("Global Bike Count Statistics")
    st.markdown("""
Initial descriptive statistics show:

- Average hourly traffic is in the dozens
- Some locations register over a thousand bikes/hour
- Strong variability justifies segmented analysis by hour and weekday
""")
    st.write(df["comptage_horaire"].describe())

    st.subheader("Average Bike Count by Weekday")
    st.markdown("""
This chart shows weekly cycling behavior:

- Traffic peaks Tuesday to Thursday
- Significant drop on Saturday, especially Sunday
- Suggests bike use is mainly for commuting
""")
    weekday_avg = df.groupby("weekday")["comptage_horaire"].mean()
    st.bar_chart(weekday_avg)

    st.subheader("Average Hourly Bike Traffic")
    st.markdown("""
Two major peaks observed:

- Morning: ~7:30 to 9:00
- Evening: ~17:30 to 19:00

Matching traditional work hours, confirming commuter use
""")
    hour_avg = df.groupby("hour")["comptage_horaire"].mean()
    st.line_chart(hour_avg)

    st.subheader("Monthly Traffic Averages")
    st.markdown("""
Seasonality is clearly visible:

- Peak months: May, June, July, September
- Lowest in winter: December to February

These changes align with weather and daylight
""")
    month_avg = df.groupby("month")["comptage_horaire"].mean()
    st.bar_chart(month_avg)

    st.subheader("Traffic: Weekday vs Weekend")
    st.markdown("""
Comparison reveals:

- Weekday: sharp peaks during work hours
- Weekend: traffic spread evenly across day

This reflects leisure/tourism use on weekends vs functional use during the week
""")
    df['type_jour'] = df['weekday'].apply(lambda x: 'weekend' if x in ['Saturday', 'Sunday'] else 'weekday')
    trafic_jour_type = df.groupby(['type_jour', 'hour'])['comptage_horaire'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=trafic_jour_type, x='hour', y='comptage_horaire', hue='type_jour', palette='Set1')
    plt.title("Traffic Comparison: Weekday vs Weekend")
    plt.xlabel("Hour")
    plt.ylabel("Bikes/hour")
    st.pyplot(plt.gcf())

    st.subheader("Top 10 Most Active Counters")
    st.markdown("""
Most active counters are on key axes:

- Boulevard de Sébastopol
- Boulevard de Ménilmontant
- Quai d'Orsay

These areas concentrate major bike traffic and are well developed for cycling
""")
    top_counters = df.groupby("nom_du_compteur")["comptage_horaire"].sum().nlargest(10)
    st.bar_chart(top_counters)

    st.subheader("Top 20 Peak Traffic Hours by Counter")
    st.markdown("""
Identifies the exact hours when each counter peaked:

- Almost all peaks occur during weekdays
- Mostly around 8:30 AM or 6:00 PM

This underlines the need for better infrastructure at peak hours
""")
    df_time_series = df.groupby(['nom_du_compteur', 'date_et_heure_de_comptage'])['comptage_horaire'].mean().unstack(level=0)
    peak_times = df_time_series.idxmax()
    peak_df = pd.DataFrame({
        'Counter': peak_times.index,
        'Peak Time': peak_times.values,
        'Max Traffic': df_time_series.max().values
    }).sort_values('Max Traffic', ascending=False).head(20)
    peak_df['Peak Time'] = pd.to_datetime(peak_df['Peak Time']).dt.strftime('%Y-%m-%d %H:%M')
    fig = px.bar(peak_df, x='Counter', y='Max Traffic', hover_data=['Peak Time'],
                 color='Max Traffic', title='Top 20 Peak Traffic Hours',
                 labels={'Max Traffic': 'Bikes/hour', 'Peak Time': 'Time'}, height=500)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    st.subheader("Hourly Traffic by Day of the Week")
    st.markdown("""
Traffic varies across weekdays. Mondays and Fridays show distinct patterns.
""")
    df_jour = df.groupby(['weekday', 'hour'])['comptage_horaire'].mean().reset_index()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_jour, x='hour', y='comptage_horaire', hue='weekday', hue_order=order, palette='viridis', linewidth=2.5)
    plt.title('Average Hourly Bike Traffic by Weekday', fontsize=16)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Bikes/Hour')
    plt.grid(alpha=0.3)
    plt.legend(title='Day', bbox_to_anchor=(1.05, 1))
    st.pyplot(plt.gcf())

    st.markdown("""
### Exploratory Analysis Conclusion
- Bike traffic follows a regular daily and weekly cycle, peaking during morning and evening commutes.
- Weekends show more consistent traffic throughout the day.
- Central roads and riversides are most used.
- The dataset is clean and well-suited for predictive modeling.
""")



# -------------------------------
# 3. Approach
# -------------------------------
elif section == "Approach":
    st.title("Modeling Approach")

    st.markdown("""
After exploring hourly bike traffic data in Paris, our objective was to forecast traffic volume and detect crowding patterns using both temporal and spatial information.

---

### Modeling Objectives

We designed two main predictive goals:

- **Bike Count Prediction (Regression)**: Forecast the number of bikes passing per hour at a given location
- **Crowding Detection (Classification)**: Detect whether a given time/location is likely to experience heavy traffic

---

### Selected Features

To capture mobility dynamics, we used the following input features:

- `hour`: Hour of the day (captures rush hour patterns)
- `weekday`: Name of the day (weekday vs weekend differences)
- `month`: Captures seasonal effects
- `latitude / longitude`: Geographic coordinates of the counter
- `counter name`: For classification, different counters have different traffic profiles

These features reflect both time-based and location-based variations in cycling traffic.

---

### Data Preprocessing

We applied the following preprocessing steps before training the models:

- **Categorical Encoding**:
    - For regression: `OneHotEncoder` on `weekday`
    - For classification: `OneHotEncoder` on `weekday` and `counter name`
- **Train/Test Split**:
    - 80% training, 20% testing
- **Feature Matrix Creation**:
    - Combined numerical and encoded categorical features using `np.hstack()`
- **Coordinate Conversion**:
    - Geographic coordinates were cast to `float32` for memory efficiency

---

### Regression Pipeline

For the regression task, we used the following process:

- Input features: `hour`, `month`, `latitude`, `longitude`, `weekday (encoded)`
- Model used: `RandomForestRegressor (n=100, random_state=42)`
- Evaluation metrics:
    - **Mean Absolute Error (MAE)**
    - **R² Score (explained variance)**
- Example input for MLflow:
    ```python
    [[8, 5, 48.8566, 2.3522, ... encoded weekday ...]]
    ```

The model was tracked using **MLflow**, and both the trained model and encoder were saved locally using `joblib`.

---

### Classification Pipeline (Crowding Detection)

To detect high-traffic conditions, we formulated a **binary classification problem**:

- A row is labeled **1 (Crowded)** if its bike count exceeds the historical average for that counter
- Labeled **0 (Low Traffic)** otherwise

Classification input features included:
- `hour`, `month`, `weekday`, `counter name`

Model used:
- `RandomForestClassifier (n=100, random_state=42)`

Evaluation metrics:
- **Confusion Matrix**
- **Accuracy**
- **Precision / Recall / F1-score**
- **ROC AUC**

The model was also saved using `joblib`, and the encoding was applied using `OneHotEncoder`.

---

This approach balances performance and interpretability, while leveraging both spatial and temporal patterns for more accurate traffic prediction and congestion detection.
""")

# -------------------------------
# 4. Modeling
# -------------------------------
elif section == "Modeling":
    st.title("Model Performance")

    model_choice = st.radio("Select model type to display:", ["Regression – Hourly Bike Count", "Classification – Crowding Detection"])

    if model_choice == "Regression – Hourly Bike Count":
        st.subheader("Regression Model – Predicting Hourly Bike Traffic")

        st.markdown("""
We trained a **RandomForestRegressor** to predict the number of bikes passing per hour at a given location.

**Features used:**
- Hour
- Month
- Latitude and Longitude of the counter
- Day of the week (encoded)

**Evaluation Metrics:**
- **Mean Absolute Error (MAE)**: average difference between predicted and actual values
- **R² Score**: proportion of variance in bike traffic explained by the model


""")
        st.subheader("Regression Results")
        col1, col2 = st.columns(2)
        col1.metric("MAE", "23.5", "↓ better")
        col2.metric("R²", "0.80", "↑ better")
        st.markdown("Model captures overall traffic trends with low error.")

    elif model_choice == "Classification – Crowding Detection":
        st.subheader("Classification Model – Crowding Detection")

        st.markdown("""
We trained a **RandomForestClassifier** to detect high-traffic (crowded) conditions for a given hour and counter.

**Features used:**
- Hour
- Month
- Day of the week
- Counter name (both encoded)

**Target variable:**
- **Crowded (1)** if the hourly count is above the counter’s historical average  
- **Low traffic (0)** otherwise

**Evaluation Metrics:**
- **Accuracy**: proportion of correct predictions  
- **Precision**: how many predicted crowded hours were correct  
- **Recall**: how many actual crowded hours were detected  
- **F1 Score**: balance between precision and recall  
- **ROC AUC**: model’s ability to distinguish crowded from non-crowded

""")
        st.subheader("Classification Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "86%")
        col2.metric("F1 Score", "0.82")
        col3.metric("ROC AUC", "0.89")
        st.markdown("Model performs well in detecting peak/crowded hours.")

# -------------------------------
# 5. Demo
# -------------------------------
elif section == "Demo":
    st.title("Live Prediction Demo")
    api_url = st.secrets.get("API_URL", "https://bike-api-111973461276.europe-west1.run.app/predict")
    mode = st.radio(
        "Prediction type:",
        ["Hourly traffic (regression)", "Crowding (classification)"],
        key="type_prediction"
    )

    st.markdown("This model predicts either the **number of bikes per hour** (regression) or a **crowding level** (classification) based on time and location inputs.")
    
    hour = st.slider("Hour of the day", 0, 23, 8)
    weekday = st.selectbox("Weekday", df["weekday"].unique())
    month = st.selectbox("Month", list(range(1, 13)))

    counters = df[['nom_du_compteur', 'latitude', 'longitude']].drop_duplicates()
    counter_name = st.selectbox("Counting location", counters['nom_du_compteur'])
    selected = counters[counters['nom_du_compteur'] == counter_name].iloc[0]
    lat = selected['latitude']
    lon = selected['longitude']

    payload = {
        "hour": hour,
        "month": month,
        "weekday": weekday,
        "lat": lat,
        "lon": lon,
        "counter_name": counter_name,
        "mode": "regression" if mode == "Hourly traffic (regression)" else "classification"
    }

    if st.button("Predict"):
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if mode == "Hourly traffic (regression)":
                pred = result.get("prediction")
                st.success(f"Prediction: {int(pred)} bikes/hour")
            else:
                pred_class = result.get("prediction")
                prob = result.get("probability", 0)
                label = "Crowded" if pred_class == 1 else "Low traffic"
                st.success(f"Prediction: {label} (probability: {prob:.2%})")

        except Exception as e:
            st.error(f"Error while contacting the API: {e}")

    st.markdown("---")
    st.caption("Prediction results provided by the deployed FastAPI backend.")

    