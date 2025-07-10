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
st.title("🔐 GCP Secret Test")

try:
    credentials_info = json.loads(st.secrets["GCP_KEY"])
    st.success("✅ Secret loaded successfully!")
    st.json(credentials_info)
except Exception as e:
    st.error("❌ Failed to load GCP_KEY from secrets.")
    st.code(str(e))

@st.cache_resource
def load_from_gcs(bucket_name, blob_name):
    try:
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        client = storage.Client(credentials=credentials, project=credentials.project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except Exception as e:
        st.error("❌ Error during GCS file download")
        st.code(str(e))
        return None

# Step 3: Load model from GCS
@st.cache_resource
def load_model_from_gcs(bucket, blob):
    file_bytes = load_from_gcs(bucket, blob)
    if file_bytes is not None:
        try:
            model = joblib.load(BytesIO(file_bytes))
            return model
        except Exception as e:
            st.error("❌ Failed to load model with joblib")
            st.code(str(e))
            return None
    return None

# Step 4: Trigger model loading via button
if st.button("🔄 Load Model from GCS"):
    st.info("⏳ Attempting to load model from GCS...")

    model = load_model_from_gcs("bike-models-bucket", "regression/model_reg.joblib")

    if model:
        st.success("✅ Model loaded successfully!")
        st.write(model)
    else:
        st.error("❌ Model loading failed. Check logs or file access.")