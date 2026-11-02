import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Market Risk Engine",
    layout="wide"
)

# --------------------------------------------------
# SHARED APPLICATION SETTINGS
# --------------------------------------------------

if "available_assets" not in st.session_state:
    stock_universe = pd.read_csv("data/stock_universe.csv")
    st.session_state.available_assets = (
        stock_universe["ticker"].dropna().astype(str).tolist()
    )

if "selected_assets" not in st.session_state:
    st.session_state.selected_assets = []

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


# --------------------------------------------------
# SHARED MARKET DATA
# --------------------------------------------------

# market_prices contains the full downloaded history (up to 10 years).
# portfolio_prices contains only the common dates in the investment period.

if "market_prices" not in st.session_state:
    st.session_state.market_prices = None

if "market_quality_report" not in st.session_state:
    st.session_state.market_quality_report = None

if "market_data_warning" not in st.session_state:
    st.session_state.market_data_warning = False

if "market_data_assets" not in st.session_state:
    st.session_state.market_data_assets = []


# --------------------------------------------------
# SHARED PORTFOLIO RESULTS
# --------------------------------------------------

if "portfolio_prices" not in st.session_state:
    st.session_state.portfolio_prices = None

if "portfolio_returns" not in st.session_state:
    st.session_state.portfolio_returns = None

if "allocation_drift" not in st.session_state:
    st.session_state.allocation_drift = None

if "portfolio_value" not in st.session_state:
    st.session_state.portfolio_value = None

if "portfolio_metrics" not in st.session_state:
    st.session_state.portfolio_metrics = None

if "portfolio_weights_decimal" not in st.session_state:
    st.session_state.portfolio_weights_decimal = None

# --------------------------------------------------
# SHARED RISK SETTINGS
# --------------------------------------------------

if "var_confidence_level" not in st.session_state:
    st.session_state.var_confidence_level = 0.95

if "var_holding_period" not in st.session_state:
    st.session_state.var_holding_period = 1

if "var_lookback_window" not in st.session_state:
    st.session_state.var_lookback_window = 252

if "var_simulations" not in st.session_state:
    st.session_state.var_simulations = 10000


# --------------------------------------------------
# RISK LIMITS
# --------------------------------------------------

if "var_limit" not in st.session_state:
    st.session_state.var_limit = 0.03

if "max_drawdown_limit" not in st.session_state:
    st.session_state.max_drawdown_limit = 0.15


# --------------------------------------------------
# APPLICATION NAVIGATION
# --------------------------------------------------

pages = {
    "Market Risk Engine": [
        st.Page("pages/0_Executive_Dashboard.py", title="Executive Dashboard"),
        st.Page("pages/1_Portfolio_Construction.py", title="Portfolio Construction"),
        st.Page("pages/2_Market_Data_Analytics.py", title="Market Data Analytics"),
        st.Page("pages/3_VaR_and_ES.py", title="VaR and ES"),
        st.Page("pages/4_Backtesting.py", title="Backtesting"),
        st.Page("pages/5_Stress_Testing.py", title="Stress Testing")
    ]
}

page = st.navigation(pages)
page.run()
