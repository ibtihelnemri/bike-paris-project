import streamlit as st
import json

st.title("🔐 GCP Secret Test")

try:
    credentials_info = json.loads(st.secrets["GCP_KEY"])
    st.success("✅ Secret loaded successfully!")
    st.json(credentials_info)
except Exception as e:
    st.error("❌ Failed to load GCP_KEY from secrets.")
    st.code(str(e))