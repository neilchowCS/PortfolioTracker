import streamlit as st
import pandas as pd
from datetime import date

from ui.fmt import fmtd, fmt_acct
from data.data_manager import (
    load_accounts, add_account, update_account, delete_account, get_account_names,
    load_transactions, add_transaction, update_transaction, delete_transaction,
    get_existing_tickers, save_transactions,
)
from data.csv_import import import_all_from_folders
from data.ticker_lookup import get_ticker_name

TAXABLE_TYPES = ["taxable", "roth_ira", "401k_pretax", "401k_roth", "401k_aftertax", "hsa"]
TXN_TYPES = ["buy", "sell", "contribution", "withdrawal", "split", "transfer"]


def _fmt(val: str) -> str:
    """Human-friendly label for dropdown values."""
    return val.replace("_", " ").title()


def _bump_data_version():
    """Increment data version so the dashboard knows to recompute."""
    st.session_state["_data_version"] = st.session_state.get("_data_version", 0) + 1


def render_accounts_panel():
    """Left column: account CRUD."""
    st.header("Accounts")
    accounts_df = load_accounts()

    # --- Add account ---
    with st.expander("Add Account", expanded=not len(accounts_df)):
        with st.form("add_account_form", clear_on_submit=True):
            new_name = st.text_input("Account Name")
            new_type = st.selectbox("Taxable Type", TAXABLE_TYPES, format_func=_fmt, key="add_acct_type")
            submitted = st.form_submit_button("Add Account")
            if submitted:
                if not new_name.strip():
                    st.error("Name cannot be empty.")
                else:
                    try:
                        add_account(new_name.strip(), new_type)
                        st.success(f"Added account '{new_name.strip()}'")
                        _bump_data_version()
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    # --- Existing accounts ---
    if not accounts_df.empty:
        st.subheader("Existing Accounts")
        for _, row in accounts_df.iterrows():
            akey = row["name"]
            with st.expander(f"{fmt_acct(row['name'])}  ({_fmt(row['taxable_type'])})"): 
                with st.form(f"edit_acct_{akey}"):
                    ed_name = st.text_input("Name", value=row["name"], key=f"ean_{akey}")
                    ed_type = st.selectbox(
                        "Taxable Type",
                        TAXABLE_TYPES,
                        index=TAXABLE_TYPES.index(row["taxable_type"]) if row["taxable_type"] in TAXABLE_TYPES else 0,
                        format_func=_fmt,
                        key=f"eat_{akey}",
                    )
                    c1, c2 = st.columns(2)
                    save = c1.form_submit_button("Save")
                    remove = c2.form_submit_button("Delete", type="secondary")
                if save:
                    try:
                        update_account(row["name"], ed_name.strip(), ed_type)
                        st.success("Updated.")
                        _bump_data_version()
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                if remove:
                    delete_account(row["name"])
                    st.success(f"Deleted '{row['name']}'")
                    _bump_data_version()
                    st.rerun()


def _ticker_input_widget(prefix: str, default: str = "") -> str:
    """Plain ticker text input.  Returns the entered symbol (uppercase).
    Ticker name is looked up lazily and stored in session state so it
    never triggers a rerun or steals focus from subsequent fields.
    """
    ticker = st.text_input("Ticker", value=default, key=f"{prefix}_ticker_input").upper().strip()
    if not ticker:
        # Clear any stale cached name
        st.session_state.pop(f"{prefix}_ticker_name", None)
        return ""
    # Show cached name instantly; schedule lookup for next rerun
    name_key = f"{prefix}_ticker_name"
    cached_ticker_key = f"{prefix}_ticker_last"
    if st.session_state.get(cached_ticker_key) != ticker:
        # Ticker changed — look up name now, cache it
        name = get_ticker_name(ticker)
        st.session_state[cached_ticker_key] = ticker
        st.session_state[name_key] = name if name and name != ticker else ""
    cached_name = st.session_state.get(name_key, "")
    if cached_name:
        st.caption(f"{ticker} — {cached_name}")
    return ticker


def render_transactions_panel():
    """Right column: transaction CRUD."""
    st.header("Transactions")
    account_names = get_account_names()

    if not account_names:
        st.info("Create an account first before adding transactions.")
        return

    # --- Reset fields after successful add ---
    if st.session_state.get("_txn_just_added"):
        st.session_state["_txn_just_added"] = False
        st.session_state["add_txn_ticker_input"] = ""
        st.session_state["add_txn_price"] = ""
        st.session_state["add_txn_shares"] = ""
        st.session_state["add_txn_amount"] = ""
        st.session_state["add_txn_split_new"] = ""
        st.session_state["add_txn_split_old"] = ""
        st.session_state.pop("add_txn_ticker_name", None)
        st.session_state.pop("add_txn_ticker_last", None)

    # --- Add transaction ---
    with st.expander("Add Transaction", expanded=True):
        txn_type = st.selectbox("Type", TXN_TYPES, format_func=_fmt, key="add_txn_type")
        acct = st.selectbox("Account", account_names, format_func=fmt_acct, key="add_txn_acct")
        txn_date = st.date_input("Date", value=date.today(), key="add_txn_date")

        ticker_val = ""
        price_val = 0.0
        shares_val = 0.0
        amount_val = 0.0

        split_new = 0.0
        split_old = 0.0

        if txn_type in ("buy", "sell"):
            ticker_val = _ticker_input_widget("add_txn")
            shares_str = st.text_input("Shares", key="add_txn_shares")
            price_str = st.text_input("Price per Share", key="add_txn_price")
            try:
                price_val = float(price_str) if price_str.strip() else 0.0
                shares_val = float(shares_str) if shares_str.strip() else 0.0
            except ValueError:
                price_val = 0.0
                shares_val = 0.0
            if price_val and shares_val:
                st.caption(f"Total: {fmtd(price_val * shares_val)}")
        elif txn_type == "split":
            ticker_val = _ticker_input_widget("add_txn")
            st.caption("Enter split ratio — e.g. for a 10-for-1 split: New = 10, Old = 1")
            sc1, sc2 = st.columns(2)
            split_new_str = sc1.text_input("New shares", key="add_txn_split_new")
            split_old_str = sc2.text_input("Old shares", key="add_txn_split_old")
            try:
                split_new = float(split_new_str) if split_new_str.strip() else 0.0
                split_old = float(split_old_str) if split_old_str.strip() else 0.0
            except ValueError:
                split_new = 0.0
                split_old = 0.0
            if split_new > 0 and split_old > 0:
                st.caption(f"Ratio: {split_new/split_old:.4g}:1 — each share becomes {split_new/split_old:.4g} shares")
        elif txn_type == "transfer":
            ticker_val = _ticker_input_widget("add_txn")
            shares_str = st.text_input("Shares", key="add_txn_shares")
            price_str = st.text_input("Cost Basis per Share", key="add_txn_price")
            try:
                price_val = float(price_str) if price_str.strip() else 0.0
                shares_val = float(shares_str) if shares_str.strip() else 0.0
            except ValueError:
                price_val = 0.0
                shares_val = 0.0
            if price_val and shares_val:
                amount_val = round(price_val * shares_val, 4)
                st.caption(f"Market value: {fmtd(amount_val)} (treated as contribution, no cash impact)")
        else:
            amount_str = st.text_input("Amount ($)", key="add_txn_amount")
            try:
                amount_val = float(amount_str) if amount_str.strip() else 0.0
            except ValueError:
                amount_val = 0.0

        if st.button("Add Transaction", key="add_txn_btn"):
            if txn_type in ("buy", "sell") and not ticker_val:
                st.error("Ticker is required for buy/sell.")
            elif txn_type in ("buy", "sell") and (price_val <= 0 or shares_val <= 0):
                st.error("Price and shares must be positive.")
            elif txn_type == "split" and not ticker_val:
                st.error("Ticker is required for a split.")
            elif txn_type == "split" and (split_new <= 0 or split_old <= 0):
                st.error("Both new and old share counts must be positive.")
            elif txn_type == "transfer" and not ticker_val:
                st.error("Ticker is required for a transfer.")
            elif txn_type == "transfer" and (price_val <= 0 or shares_val <= 0):
                st.error("Cost basis and shares must be positive.")
            elif txn_type in ("contribution", "withdrawal") and amount_val <= 0:
                st.error("Amount must be positive.")
            else:
                if txn_type == "transfer":
                    add_transaction(
                        txn_type="transfer",
                        account=acct,
                        ticker=ticker_val,
                        price=price_val,
                        shares=shares_val,
                        amount=round(price_val * shares_val, 4),
                        date=txn_date.isoformat(),
                    )
                elif txn_type == "split":
                    # Store ratio as shares (new/old), price holds raw new:old for display
                    add_transaction(
                        txn_type="split",
                        account=acct,
                        ticker=ticker_val,
                        price=split_old,
                        shares=split_new,
                        amount=0.0,
                        date=txn_date.isoformat(),
                    )
                else:
                    add_transaction(
                        txn_type=txn_type,
                        account=acct,
                        ticker=ticker_val,
                        price=price_val,
                        shares=shares_val,
                        amount=amount_val,
                        date=txn_date.isoformat(),
                    )
                st.session_state["_txn_just_added"] = True
                st.success("Transaction added.")
                _bump_data_version()
                st.rerun()

    # --- Transaction history ---
    txns = load_transactions()
    if txns.empty:
        return

    st.subheader("Transaction History")
    filter_options = ["All"] + account_names
    # Reset filter if previously selected account no longer exists
    stored_filter = st.session_state.get("txn_filter_acct")
    if stored_filter and stored_filter not in filter_options:
        st.session_state.pop("txn_filter_acct", None)
    filter_acct = st.selectbox("Filter by Account", filter_options, format_func=lambda x: "All" if x == "All" else fmt_acct(x), key="txn_filter_acct")
    display = txns if filter_acct == "All" else txns[txns["account"] == filter_acct]
    display = display.sort_values("date", ascending=False).reset_index(drop=True)

    # Styling: imported rows get a muted background
    _IMPORTED_BG = "background-color: rgba(100,100,100,0.08);"
    _MANUAL_BG = ""

    # Check if we're editing a row
    editing_id = st.session_state.get("_editing_txn_id")

    # ── Pagination ──────────────────────────────────────────────────────
    PAGE_SIZE = 25
    total_rows = len(display)
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
    if total_rows > PAGE_SIZE:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1,
                               key="txn_page", step=1)
        st.caption(f"Showing {min(PAGE_SIZE, total_rows - (page-1)*PAGE_SIZE)} of {total_rows} transactions (page {page}/{total_pages})")
    else:
        page = 1
    page_display = display.iloc[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]

    # ── Styled table with edit buttons ────────────────────────────────
    # Header
    hdr = st.columns([0.5, 1, 1, 1.2, 0.8, 1, 1, 1, 0.6])
    for col, label in zip(hdr, ["ID", "Date", "Type", "Account", "Ticker", "Price", "Shares", "Amount", ""]):
        col.markdown(f"**{label}**")

    def _fmt_num(v, fmt_str):
        try:
            fv = float(v)
            if fv != fv or fv == 0:  # nan or zero
                return "—"
            return fmt_str.format(fv)
        except (ValueError, TypeError):
            return "—"

    def _safe_float(v):
        try:
            f = float(v)
            return f if f == f else 0.0  # reject nan
        except (ValueError, TypeError):
            return 0.0

    for _, row in page_display.iterrows():
        is_imported = str(row.get("source", "manual")) != "manual"
        rid = int(row["id"])
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d") if pd.notna(row["date"]) else ""
        ticker = str(row["ticker"]) if pd.notna(row["ticker"]) and row["ticker"] != "" else "—"
        price = _fmt_num(row.get("price", ""), "{:,.4f}")
        shares = _fmt_num(row.get("shares", ""), "{:,.4f}")
        amount = _fmt_num(row.get("amount", ""), "${:,.2f}")

        if is_imported:
            container = st.container()
            container.markdown(
                f'<div style="{_IMPORTED_BG} padding:2px 6px; border-radius:4px; margin:-4px 0;">',
                unsafe_allow_html=True,
            )
        else:
            container = st

        cols = container.columns([0.5, 1, 1, 1.2, 0.8, 1, 1, 1, 0.6])
        cols[0].text(str(rid))
        cols[1].text(date_str)
        cols[2].text(_fmt(row["type"]))
        cols[3].text(fmt_acct(str(row["account"])))
        cols[4].text(ticker)
        cols[5].text(price)
        cols[6].text(shares)
        cols[7].text(amount)

        if is_imported:
            cols[8].markdown(f":lock:", help=f"Imported from {row.get('source','')}")
            container.markdown("</div>", unsafe_allow_html=True)
        else:
            if cols[8].button("✏️", key=f"edit_btn_{rid}", help="Edit this transaction"):
                st.session_state["_editing_txn_id"] = rid
                # Pre-fill edit form values
                st.session_state["_edit_type"] = row["type"]
                st.session_state["_edit_acct"] = str(row["account"])
                st.session_state["_edit_date"] = pd.to_datetime(row["date"]).date() if pd.notna(row["date"]) else date.today()
                st.session_state["_edit_ticker"] = str(row["ticker"]) if pd.notna(row["ticker"]) and row["ticker"] != "" else ""
                st.session_state["_edit_price"] = _safe_float(row.get("price", 0))
                st.session_state["_edit_shares"] = _safe_float(row.get("shares", 0))
                st.session_state["_edit_amount"] = _safe_float(row.get("amount", 0))
                st.rerun()

    # ── Inline edit form ──────────────────────────────────────────────
    if editing_id is not None:
        sel_match = txns[txns["id"] == editing_id]
        if sel_match.empty:
            st.session_state.pop("_editing_txn_id", None)
            return

        st.divider()
        st.markdown(f"**Editing Transaction #{editing_id}**")

        # Read pre-filled values from session state
        prefill_type = st.session_state.get("_edit_type", "buy")
        prefill_acct = st.session_state.get("_edit_acct", account_names[0])
        prefill_date = st.session_state.get("_edit_date", date.today())
        prefill_ticker = st.session_state.get("_edit_ticker", "")
        prefill_price = st.session_state.get("_edit_price", 0.0)
        prefill_shares = st.session_state.get("_edit_shares", 0.0)
        prefill_amount = st.session_state.get("_edit_amount", 0.0)

        e_type = st.selectbox(
            "Type", TXN_TYPES,
            index=TXN_TYPES.index(prefill_type) if prefill_type in TXN_TYPES else 0,
            format_func=_fmt, key="e_txn_type",
        )
        e_acct = st.selectbox(
            "Account", account_names,
            index=account_names.index(prefill_acct) if prefill_acct in account_names else 0,
            format_func=fmt_acct,
            key="e_txn_acct",
        )
        e_date = st.date_input("Date", value=prefill_date, key="e_txn_date")

        e_ticker = ""
        e_price = 0.0
        e_shares = 0.0
        e_amount = 0.0
        e_split_new = 0.0
        e_split_old = 0.0

        if e_type in ("buy", "sell"):
            e_ticker = _ticker_input_widget("e_txn", default=prefill_ticker)
            e_shares = st.number_input("Shares", min_value=0.0, value=prefill_shares,
                                       format="%.4f", key="e_txn_shares")
            e_price = st.number_input("Price per Share", min_value=0.0, value=prefill_price,
                                      format="%.4f", key="e_txn_price")
        elif e_type == "split":
            e_ticker = _ticker_input_widget("e_txn", default=prefill_ticker)
            st.caption("Split ratio — New : Old")
            esc1, esc2 = st.columns(2)
            e_split_new = esc1.number_input("New shares", min_value=0.0, value=prefill_shares,
                                            format="%.4f", key="e_txn_split_new")
            e_split_old = esc2.number_input("Old shares", min_value=0.0, value=prefill_price,
                                            format="%.4f", key="e_txn_split_old")
        elif e_type == "transfer":
            e_ticker = _ticker_input_widget("e_txn", default=prefill_ticker)
            e_shares = st.number_input("Shares", min_value=0.0, value=prefill_shares,
                                       format="%.4f", key="e_txn_shares")
            e_price = st.number_input("Cost Basis per Share", min_value=0.0, value=prefill_price,
                                      format="%.4f", key="e_txn_price")
            e_amount = round(e_price * e_shares, 4)
        else:
            e_amount = st.number_input("Amount ($)", min_value=0.0, value=prefill_amount,
                                       format="%.2f", key="e_txn_amount")

        ec1, ec2, ec3 = st.columns(3)
        if ec1.button("Save Changes", key="e_txn_save"):
            if e_type == "split":
                update_transaction(txn_id=editing_id, txn_type=e_type, account=e_acct,
                                   ticker=e_ticker, price=e_split_old, shares=e_split_new,
                                   amount=0.0, date=e_date.isoformat())
            else:
                update_transaction(txn_id=editing_id, txn_type=e_type, account=e_acct,
                                   ticker=e_ticker, price=e_price, shares=e_shares,
                                   amount=e_amount, date=e_date.isoformat())
            st.session_state.pop("_editing_txn_id", None)
            st.success("Updated.")
            _bump_data_version()
            st.rerun()
        if ec2.button("Delete", key="e_txn_del", type="secondary"):
            delete_transaction(editing_id)
            st.session_state.pop("_editing_txn_id", None)
            st.success("Deleted.")
            _bump_data_version()
            st.rerun()
        if ec3.button("Cancel", key="e_txn_cancel"):
            st.session_state.pop("_editing_txn_id", None)
            st.rerun()


def _render_csv_import():
    """CSV import section — scan imports/schwab, imports/fidelity, imports/etrade folders."""
    st.header("Import from Brokerage CSV")
    st.caption(
        "Place CSV exports into the appropriate folder:\n"
        "- `imports/schwab/`\n"
        "- `imports/fidelity/`\n"
        "- `imports/etrade/`\n\n"
        "Filename (without .csv) = account name. Imported rows are read-only."
    )

    all_rows, all_skipped = import_all_from_folders()

    if not all_rows:
        st.info("No CSV files found in imports/ folders.")
        return

    account_names = get_account_names()
    total_parsed = sum(len(r) for r in all_rows.values())
    total_skipped = sum(len(s) for s in all_skipped.values())

    def _acct_from_key(key: str) -> str:
        """Extract account name from broker/filename.csv key."""
        basename = key.split("/", 1)[-1]  # drop broker prefix
        return basename.rsplit(".", 1)[0]  # drop .csv

    # Show summary per file
    for fname in sorted(all_rows.keys()):
        parsed = all_rows[fname]
        skipped = all_skipped.get(fname, [])
        acct_name = _acct_from_key(fname)

        with st.expander(f"**{fname}** → `{fmt_acct(acct_name)}` — {len(parsed)} rows, {len(skipped)} skipped"):
            if parsed:
                preview = pd.DataFrame(parsed)
                st.dataframe(preview, use_container_width=True, hide_index=True)
            if skipped:
                st.markdown("**Skipped:**")
                for s in skipped:
                    st.caption(f"line {s['line']}: {s['reason']}")

    # Show files needing new accounts
    new_accounts = {}
    for fname in sorted(all_rows.keys()):
        acct_name = _acct_from_key(fname)
        if acct_name not in account_names and all_rows[fname]:
            new_accounts[fname] = acct_name

    if new_accounts:
        st.subheader("New accounts to create")
        for fname, acct_name in new_accounts.items():
            new_accounts[fname] = st.selectbox(
                f"Account type for '{fmt_acct(acct_name)}'",
                TAXABLE_TYPES, format_func=_fmt,
                key=f"csv_acct_type_{fname}",
            )

    if total_skipped > 0:
        st.warning(f"⚠️ {total_skipped} row(s) skipped across {sum(1 for s in all_skipped.values() if s)} file(s). "
                   "Expand files above to see details.")

    st.markdown(f"**Total: {total_parsed} transactions** from {len(all_rows)} files")

    if total_parsed > 0 and st.button("Import All", key="csv_import_all"):
        existing = load_transactions()
        # Wipe all previously imported rows (keep manual only)
        if "source" in existing.columns:
            existing = existing[existing["source"] == "manual"]
        next_id = int(existing["id"].max()) + 1 if not existing.empty else 1
        imported_count = 0

        for fname in sorted(all_rows.keys()):
            parsed = all_rows[fname]
            if not parsed:
                continue
            acct_name = _acct_from_key(fname)
            source_tag = fname

            # Create account if needed
            if acct_name not in get_account_names():
                acct_type_key = f"csv_acct_type_{fname}"
                acct_type = st.session_state.get(acct_type_key, "taxable")
                add_account(acct_name, acct_type)

            # Build new rows
            new_rows = []
            for r in parsed:
                row = {
                    "id": next_id,
                    "type": r["type"],
                    "ticker": r.get("ticker", ""),
                    "price": r.get("price", 0),
                    "shares": r.get("shares", 0),
                    "amount": r.get("amount", 0),
                    "date": r["date"],
                    "account": acct_name,
                    "source": source_tag,
                }
                new_rows.append(row)
                next_id += 1

            new_df = pd.DataFrame(new_rows)
            existing = pd.concat([existing, new_df], ignore_index=True)
            imported_count += len(new_rows)

        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        existing["id"] = existing["id"].astype(int)
        save_transactions(existing)
        st.success(f"Imported {imported_count} transactions from {len(all_rows)} files")
        # Clear stale transaction UI state
        for key in list(st.session_state.keys()):
            if key.startswith(("_editing_txn_id", "_edit_", "edit_btn_", "txn_page", "txn_filter_acct")):
                st.session_state.pop(key, None)
        _bump_data_version()
        st.rerun()


def render():
    """Main entry point for the Data Entry tab."""
    acct_col, txn_col = st.columns(2)
    with acct_col:
        render_accounts_panel()
        _render_csv_import()
    with txn_col:
        render_transactions_panel()
