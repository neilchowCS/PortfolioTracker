import streamlit as st

from ui import data_entry, dashboard, portfolio_overview, rebalance, retirement

st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# Clear stale caches on app startup — version bump forces re-clear after code changes
_CACHE_VERSION = 6
if st.session_state.get("_cache_version") != _CACHE_VERSION:
    st.cache_data.clear()
    st.session_state["_cache_version"] = _CACHE_VERSION

tab_entry, tab_display, tab_overview, tab_rebalance, tab_retire = st.tabs(
    ["Data Entry", "Dashboard", "Portfolio", "Rebalance", "Retirement"]
)

with tab_entry:
    data_entry.render()

with tab_display:
    dashboard.render()

with tab_overview:
    portfolio_overview.render()

with tab_rebalance:
    rebalance.render()

with tab_retire:
    retirement.render()
