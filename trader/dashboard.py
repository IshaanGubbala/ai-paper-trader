"""Streamlit dashboard — live portfolio + interactive backtester."""
import threading
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from trader.config import ASSETS, STARTING_CAPITAL
from trader.state.store import Store

st.set_page_config(page_title="AI Paper Trader", layout="wide")

# ── Shared store ────────────────────────────────────────────────────────────
@st.cache_resource
def get_store() -> Store:
    s = Store()
    s.initialize()
    return s


store = get_store()


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("AI Paper Trader")
page = st.sidebar.radio("View", ["Portfolio", "Theses", "Trade Log", "Backtest"])


# ── Portfolio page ───────────────────────────────────────────────────────────
if page == "Portfolio":
    st.title("Live Portfolio")
    p = store.get_portfolio()
    if p is None:
        st.info("No portfolio saved yet. Start the scheduler to begin trading.")
    else:
        cash = p["cash"]
        positions = p["positions"]
        pos_value = sum(pos["shares"] * pos["entry_price"] for pos in positions)
        total = cash + pos_value

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Value", f"${total:,.2f}")
        c2.metric("Cash", f"${cash:,.2f}")
        c3.metric("Invested", f"${pos_value:,.2f}")

        if positions:
            st.subheader("Open Positions")
            st.dataframe(
                pd.DataFrame(positions)[
                    ["symbol", "shares", "entry_price", "take_profit_price",
                     "stop_loss_price", "confidence", "reason"]
                ],
                use_container_width=True,
            )
        else:
            st.info("No open positions.")

        snaps = store.get_snapshots()
        if snaps:
            st.subheader("Equity Curve")
            eq_df = pd.DataFrame(snaps)
            eq_df["recorded_at"] = pd.to_datetime(eq_df["recorded_at"])
            eq_df = eq_df.set_index("recorded_at")
            st.line_chart(eq_df["value"])


# ── Theses page ──────────────────────────────────────────────────────────────
elif page == "Theses":
    st.title("Weekly LLM Theses")
    rows = store.get_all_thesis()
    if not rows:
        st.info("No theses yet. The scheduler generates them every Monday at 6 AM ET.")
    else:
        df = pd.DataFrame(rows)
        # Colour stance column
        def colour_stance(val):
            c = {"bullish": "green", "bearish": "red", "neutral": "gray"}.get(val, "black")
            return f"color: {c}; font-weight: bold"
        styled = df[["asset", "stance", "confidence", "reasoning", "horizon", "updated_at"]].style.applymap(
            colour_stance, subset=["stance"]
        )
        st.dataframe(styled, use_container_width=True)


# ── Trade log page ───────────────────────────────────────────────────────────
elif page == "Trade Log":
    st.title("Trade Log")
    trades = store.get_trades()
    if not trades:
        st.info("No trades recorded yet.")
    else:
        df = pd.DataFrame(trades)
        df["executed_at"] = pd.to_datetime(df["executed_at"])
        st.dataframe(
            df[["symbol", "action", "shares", "price", "reason", "executed_at"]]
            .sort_values("executed_at", ascending=False),
            use_container_width=True,
        )


# ── Backtest page ─────────────────────────────────────────────────────────────
elif page == "Backtest":
    st.title("Backtest")

    # ── Controls ──
    with st.form("bt_form"):
        c1, c2, c3 = st.columns(3)
        start = c1.date_input("Start date", value=date.today() - timedelta(days=365))
        end = c2.date_input("End date", value=date.today() - timedelta(days=1))
        capital = c3.number_input("Starting capital ($)", value=100_000, step=10_000)
        run_id = st.text_input("Run ID (for caching)", value="dashboard_run")
        submitted = st.form_submit_button("Run Backtest")

    if submitted:
        if start >= end:
            st.error("Start date must be before end date.")
        else:
            st.session_state["bt_running"] = True
            st.session_state["bt_progress"] = None
            st.session_state["bt_result"] = None
            st.session_state["bt_error"] = None

            progress_placeholder = st.empty()
            status_bar = st.progress(0)

            def _run_bt():
                from trader.backtest.runner import run_backtest
                total_days = (end - start).days

                def on_progress(current_date):
                    elapsed = (current_date - start).days
                    pct = min(elapsed / max(total_days, 1), 1.0)
                    st.session_state["bt_progress"] = (current_date, pct)

                try:
                    result = run_backtest(
                        assets=ASSETS,
                        start_date=start,
                        end_date=end,
                        starting_capital=float(capital),
                        run_id=run_id,
                        progress_callback=on_progress,
                    )
                    st.session_state["bt_result"] = result
                except Exception as e:
                    st.session_state["bt_error"] = str(e)
                finally:
                    st.session_state["bt_running"] = False

            thread = threading.Thread(target=_run_bt, daemon=True)
            thread.start()

            # Poll while running
            while st.session_state.get("bt_running"):
                prog = st.session_state.get("bt_progress")
                if prog:
                    cur_date, pct = prog
                    progress_placeholder.text(f"Simulating {cur_date}…")
                    status_bar.progress(pct)
                thread.join(timeout=0.5)

            status_bar.progress(1.0)
            progress_placeholder.empty()

    # ── Results ──
    result = st.session_state.get("bt_result")
    error = st.session_state.get("bt_error")

    if error:
        st.error(f"Backtest failed: {error}")

    if result:
        st.subheader("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{result['total_return_pct']:.1f}%")
        m2.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
        m3.metric("Max Drawdown", f"{result['max_drawdown_pct']:.1f}%")
        m4.metric("Win Rate", f"{result['win_rate']:.1f}%")

        eq = result.get("equity_curve", [])
        if eq:
            st.subheader("Equity Curve")
            st.line_chart(pd.Series(eq, name="Portfolio Value"))

        trades = result.get("trades", [])
        if trades:
            st.subheader(f"Trades ({len(trades)})")
            df = pd.DataFrame(trades)
            df["pnl"] = df["pnl"].round(2)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No trades were executed in this backtest period.")
