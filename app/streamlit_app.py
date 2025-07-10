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
    #credentials = service_account.Credentials.from_service_account_file("app/key-gcp.json")
    credentials_info = json.loads(st.secrets["GCP_KEY"])
    #credentials_info = dict(st.secrets["GCP_KEY"])
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = storage.Client(credentials=credentials, project=credentials.project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

@st.cache_resource
def load_model_from_gcs(bucket, blob):
    return joblib.load(BytesIO(load_from_gcs(bucket, blob)))

def load_all_data_and_models():
    try:
        ...
        print("✅ Starting to load models")
        model_reg = load_model_from_gcs("bike-models-bucket", "regression/model_reg.joblib")
        print("✅ Regression model loaded")
        ...
    except Exception as e:
        print("❌ Failed to load models")
        print(e)
        raise e  # or st.error(str(e)) if inside app