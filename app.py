import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from engine import (
    simulate_gold_prices, ohlc_from_close, ema,
    pa_signals, backtest, metrics, summarize_trades
)

st.set_page_config(page_title="Gold PA Backtester", layout="wide")

st.title("Gold price action backtester")

with st.sidebar:
    st.subheader("Simulation")
    n_days = st.number_input("Days", 100, 3000, 900, 50)
    start_price = st.number_input("Start price", 100.0, 10000.0, 2000.0, 50.0)
    drift = st.number_input("Annual drift", 0.0, 0.50, 0.03, 0.01)
    vol = st.number_input("Annual vol", 0.01, 1.00, 0.16, 0.01)
    seed = st.number_input("Seed", 0, 10_000, 7, 1)

    st.subheader("Signals")
    ema_fast = st.number_input("EMA fast", 5, 200, 21, 1)
    ema_trend = st.number_input("EMA trend", 10, 400, 50, 1)
    min_pb = st.number_input("Min pullback bars", 1, 20, 2, 1)
    fail_win = st.number_input("Fail window days", 1, 60, 6, 1)
    second_win = st.number_input("Second entry window days", 1, 60, 12, 1)
    vol_lb = st.number_input("Volume lookback", 5, 200, 20, 1)
    atr_lb = st.number_input("ATR lookback", 5, 200, 20, 1)
    vol_drop = st.number_input("Volume drop multiplier", 0.5, 1.5, 0.9, 0.05)
    atr_drop = st.number_input("ATR drop multiplier", 0.5, 1.5, 0.9, 0.05)
    allow_shorts = st.checkbox("Allow shorts", value=False)

    st.subheader("Risk")
    start_cash = st.number_input("Start cash", 1000.0, 1_000_000.0, 10_000.0, 500.0)
    risk_pt = st.number_input("Risk per trade", 0.001, 0.05, 0.01, 0.001)
    stop_atr = st.number_input("Stop ATR mult", 0.5, 10.0, 2.0, 0.1)
    tp_atr = st.number_input("Take profit ATR mult", 0.5, 20.0, 4.0, 0.1)
    fee_bps = st.number_input("Fee bps", 0.0, 50.0, 2.0, 0.5)
    slip_bps = st.number_input("Slippage bps", 0.0, 50.0, 1.0, 0.5)

run = st.button("Run backtest", type="primary")

if run:
    close_series = simulate_gold_prices(
        n_days=int(n_days),
        start_price=float(start_price),
        annual_drift=float(drift),
        annual_vol=float(vol),
        seed=int(seed)
    )
    df = ohlc_from_close(close_series)

    sig = pa_signals(
        df,
        ema_fast_len=int(ema_fast),
        ema_trend_len=int(ema_trend),
        min_pullback_bars=int(min_pb),
        fail_window=int(fail_win),
        second_entry_window=int(second_win),
        vol_lookback=int(vol_lb),
        atr_lookback=int(atr_lb),
        vol_drop_mult=float(vol_drop),
        atr_drop_mult=float(atr_drop),
        allow_shorts=bool(allow_shorts)
    )

    equity, trades = backtest(
        df_ohlc=df,
        signals_df=sig,
        start_cash=float(start_cash),
        risk_per_trade=float(risk_pt),
        stop_atr_mult=float(stop_atr),
        tp_atr_mult=float(tp_atr),
        fee_bps=float(fee_bps),
        slippage_bps=float(slip_bps),
        allow_shorts=bool(allow_shorts)
    )

    m = metrics(equity)
    ts = summarize_trades(trades)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total return", f"{m['Total Return']:.2%}")
        st.metric("CAGR", f"{m['CAGR']:.2%}" if pd.notna(m['CAGR']) else "n a")
    with col2:
        st.metric("Sharpe", f"{m['Sharpe']:.2f}")
        st.metric("Max drawdown", f"{m['Max Drawdown']:.2%}")

    st.write(f"Trades count: {ts.get('trades', 0)}  Win rate: {ts.get('win_rate', 0):.1%}" if ts.get('trades',0) > 0 else "No trades")

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(df.index, df["Close"], label="Close")
    ax1.plot(df.index, ema(df["Close"], int(ema_fast)), label=f"EMA{int(ema_fast)}", alpha=0.8)
    ax1.plot(df.index, ema(df["Close"], int(ema_trend)), label=f"EMA{int(ema_trend)}", alpha=0.8)
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(equity.index, equity.values, label="Equity", alpha=0.7)
    ax2.set_ylabel("Equity")
    fig.tight_layout()
    st.pyplot(fig)

    if not trades.empty:
        st.dataframe(trades.tail(100))
        csv = trades.to_csv(index=False).encode()
        st.download_button("Download trades CSV", csv, "trades.csv", "text/csv")
