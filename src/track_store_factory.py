"""Builds a SheetsStore from Streamlit secrets.

Kept out of track_logger.py so that module stays free of any
streamlit/gspread/secrets coupling and importable in tests/CI without
those packages. The imports are lazy for the same reason.
"""
from __future__ import annotations

from src.track_logger import SheetsStore


def make_sheets_store_from_secrets() -> SheetsStore:
    import gspread
    import streamlit as st

    sa_info = dict(st.secrets["gcp_service_account"])
    tr = st.secrets["track_record"]
    gc = gspread.service_account_from_dict(sa_info)
    ws = gc.open_by_key(tr["sheet_id"]).worksheet(tr["worksheet_name"])
    return SheetsStore(ws)
