import streamlit as st

st.set_page_config(page_title="Upload Data", layout="wide")

st.title("Upload & Preprocess Data")
st.write("This page is for uploading and preparing transaction data.")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload transactions file (.xlsx or .csv)",
    type=["xlsx", "csv"]
)

if uploaded_file:
    st.success("File uploaded successfully!")

st.markdown("---")

st.markdown("---")

if st.button("← Back to Analytics Dashboard"):
    st.switch_page("app1.py")
