import streamlit as st
import pandas as pd

# --------------------------------------------------
# SHARED APPLICATION STATE
# --------------------------------------------------

if "available_assets" not in st.session_state:
    st.session_state.available_assets = (
        pd.read_csv("data/stock_universe.csv")["ticker"].tolist()
    )

if "selected_assets" not in st.session_state:
    st.session_state.selected_assets = ["GOOGL", "WMT"]

if "start_date" not in st.session_state:
    st.session_state.start_date = pd.Timestamp("2020-01-01").date()

if "end_date" not in st.session_state:
    st.session_state.end_date = (pd.Timestamp.today() - pd.Timedelta(days=5)).date()

if "portfolio_weights" not in st.session_state:
    st.session_state.portfolio_weights = {}

if "initial_investment" not in st.session_state:
    st.session_state.initial_investment = 10000.0

if "rebalance_frequency" not in st.session_state:
    st.session_state.rebalance_frequency = "Buy and Hold"