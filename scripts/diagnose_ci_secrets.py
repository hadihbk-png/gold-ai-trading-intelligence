"""TEMPORARY DIAGNOSTIC — read-only Sheets connectivity check for CI.

Mirrors make_sheets_store_from_secrets() exactly: reads Streamlit secrets
([gcp_service_account] + [track_record] from .streamlit/secrets.toml), builds a
gspread client, and attempts to open the configured worksheet.

Read-only: no append, no update_outcome, no chain writes — it never mutates the
ledger. Prints redacted identifiers only: never the private key, never the full
sheet_id. Safe to delete once CI credentials are confirmed working.
"""

from __future__ import annotations

import sys


def main() -> int:
    import gspread
    import streamlit as st

    # Same source as make_sheets_store_from_secrets().
    sa_info = dict(st.secrets["gcp_service_account"])
    tr = st.secrets["track_record"]
    sheet_id = str(tr["sheet_id"])
    worksheet_name = str(tr["worksheet_name"])

    # Redacted identifiers only — no private key, no full sheet_id.
    print("SA_EMAIL = " + str(sa_info.get("client_email", "")))
    print("SHEET_ID_LEN = " + str(len(sheet_id)))
    print("SHEET_ID_TAIL = " + sheet_id[-4:])
    print("WORKSHEET = " + worksheet_name)

    try:
        gc = gspread.service_account_from_dict(sa_info)
        sh = gc.open_by_key(sheet_id)
        sh.worksheet(worksheet_name)          # resolve the tab (read-only)
        print("CI CONNECT OK -> " + sh.title)
        return 0
    except Exception as exc:                  # noqa: BLE001 — diagnostic surfaces any failure
        # repr only; gspread/google-auth exceptions do not embed the private key.
        print("CI CONNECT FAIL: " + repr(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
