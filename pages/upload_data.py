import streamlit as st

st.set_page_config(page_title="Upload Data", layout="wide")

st.title("Upload & Preprocess Data")

st.write("This page is for uploading and preparing transaction data.")

st.markdown("---")

if st.button("⬅ Back to Analytics Dashboard"):
    st.switch_page("app1")
