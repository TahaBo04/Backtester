import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from engine import (
    simulate_gold_prices,
    simulate_gold_prices_5m,
    ohlc_from_close,
    ema,
    pa_signals,
    backtest,
    metrics,
    summarize_trades
)

st.set_page_config(page_title="Gold PA Backtester", layout="wide")
st.title("Gold price action backtester")

with st.sidebar:
    tf = st.selectbox("Timeframe", ["5-Min", "Daily"])
    if tf == "5-Min":
        n_bars = st.number_input("Number of 5-min bars", 200, 50000, 5000, 100)
    else:
        n_days = st.number_input("Number of business days", 50, 3000, 900, 50)

    st.subheader("Simulation")
    start_price = st.number_input("Start price", 100.0, 10000.0, 2000.0, 50.0)
    drift = st.number_input("Annual drift", 0.0, 0.50, 0.03, 0.01)
    vol = st.number_input("Annual vol", 0.01, 1.00, 0.16, 0.01)
    seed = st.number_input("Seed", 0, 10_000, 7, 1)

    st.subheader("Signals")
    ema_fast = st.number_input("EMA fast", 5, 200, 21, 1)
    ema_trend = st.number_input("EMA trend", 10, 400, 50, 1)
    ema_opp   = st.number_input("Opposing EMA length", 3, 21, 9, 1)
    min_pb    = st.number_input("Min pullback bars", 1, 20, 2, 1)
    fail_win  = st.number_input("Fail window bars", 1, 60, 6, 1)
    second_win= st.number_input("Second entry window bars", 1, 60, 12, 1)
    vol_lb    = st.number_input("Volume lookback", 5, 200, 20, 1)
    atr_lb    = st.number_input("ATR lookback", 5, 200, 20, 1)
    slow_mult = st.number_input("Volatility slow multiplier ATR/ATR_SMA", 0.60, 1.00, 0.90, 0.01)
    allow_shorts = st.checkbox("Allow shorts", value=False)

    st.subheader("Risk")
    start_cash = st.number_input("Start cash", 1000.0, 1_000_000.0, 10_000.0, 500.0)
    fixed_cash = st.number_input("Cash per trade USD", 10.0, 500.0, 50.0, 5.0)
    tp_base    = st.number_input("Base TP USD", 5.0, 30.0, 10.0, 1.0)
    tp_min     = st.number_input("Min TP USD", 1.0, 30.0, 7.0, 1.0)
    tp_max     = st.number_input("Max TP USD", 5.0, 50.0, 15.0, 1.0)
    sl_frac    = st.number_input("SL fraction of TP", 0.10, 0.50, 0.45, 0.05)
    sl_frac_pb = st.number_input("SL fraction pullbacks", 0.10, 0.50, 0.35, 0.05)
    fee_bps    = st.number_input("Fee bps", 0.0, 50.0, 2.0, 0.5)
    slip_bps   = st.number_input("Slippage bps", 0.0, 50.0, 1.0, 0.5)
    run = st.button("Run backtest", type="primary")

if run:
    if tf == "5-Min":
        close_series = simulate_gold_prices_5m(
            n_bars=int(n_bars),
            start_price=float(start_price),
            annual_drift=float(drift),
            annual_vol=float(vol),
            seed=int(seed),
            freq="5min"
        )
    else:
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
    ema_opposing_len=int(ema_opp),
    min_pullback_bars=int(min_pb),
    fail_window=int(fail_win),
    second_entry_window=int(second_win),
    vol_lookback=int(vol_lb),
    atr_lookback=int(atr_lb),
    slow_mult=float(slow_mult),
    allow_shorts=bool(allow_shorts)
    )

    equity, trades = backtest(
    df_ohlc=df,
    signals_df=sig,
    start_cash=float(start_cash),
    fixed_position_cash=float(fixed_cash),
    tp_usd_min=float(tp_min),
    tp_usd_max=float(tp_max),
    tp_usd_base=float(tp_base),
    sl_frac=float(sl_frac),
    sl_frac_pullback=float(sl_frac_pb),
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

    # Candlestick with EMAs overlay
    ap = [
        mpf.make_addplot(ema(df["Close"], int(ema_fast)), panel=0),
        mpf.make_addplot(ema(df["Close"], int(ema_trend)), panel=0)
    ]
    fig, _ = mpf.plot(
        df,
        type="candle",
        addplot=ap,
        volume=True,
        returnfig=True,
        figsize=(12, 6)
    )
    st.pyplot(fig)

    if not trades.empty:
        st.dataframe(trades.tail(200))
        csv = trades.to_csv(index=False).encode()
        st.download_button("Download trades CSV", csv, "trades.csv", "text/csv")
