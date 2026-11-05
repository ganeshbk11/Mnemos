import tempfile
import streamlit as st
from ingest import ingest
from main import answer

st.title("Manual Q&A")

uploaded = st.file_uploader("Upload your manual", type="pdf")

if uploaded and st.session_state.get("name") != uploaded.name:
    with st.spinner("Reading your manual..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded.read())
        st.session_state.store = ingest(f.name)
        st.session_state.name = uploaded.name

if "store" in st.session_state:
    question = st.text_input("Ask a question about your manual")
    if question:
        st.write(answer(question, st.session_state.store))