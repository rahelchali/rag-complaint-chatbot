import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="CrediTrust Intelligent Complaint RAG Portal", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🛡️ CrediTrust Financial — AI Intelligent Complaint RAG Portal")
st.markdown("#### Transforming Raw Unstructured Customer Feedback into Actionable Strategic Product Insights")
st.markdown("---")

st.sidebar.header("⚙️ RAG Engine Parameters")
product_filter = st.sidebar.selectbox(
    "Target Product Vertical Filter",
    ["All Products", "Credit Cards", "Personal Loans", "Savings Accounts", "Money Transfers"]