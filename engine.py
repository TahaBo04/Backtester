import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- simulation ----------
def simulate_gold_prices(n_days=750, start_price=2000.0, annual_drift=0.02, annual_vol=0.15, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = annual_drift
    sigma = annual_vol
    shocks = rng.normal(loc=0.0, scale=np.sqrt(dt), size=n_days)
    log_rets = (mu - 0.5 * sigma**2) * dt + sigma * shocks
    prices = start_price * np.exp(np.cumsum(log_rets))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.Series(prices, index=idx, name="Close")

def ohlc_from_close(close: pd.Series, seed=123):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=close.index)
    df["Close"] = close
    vol_proxy = close.pct_change().rolling(20).std().bfill()
    vol_proxy = vol_proxy.replace(0, vol_proxy[vol_proxy > 0].min())
    base_range = (close * vol_proxy * 0.6).clip(lower=0.2)
    up = rng.uniform(0.2, 1.0, size=len(df))
    down = rng.uniform(0.2, 1.0, size=len(df))
    open_price = df["Close"].shift(1).fillna(df["Close"].iloc[0])
    high = np.maximum(df["Close"], open_price) + up * base_range.values
    low = np.minimum(df["Close"], open_price) - down * base_range.values
    df["Open"] = open_price
    df["High"] = high
    df["Low"] = low
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"]  = df[["Low", "Open", "Close"]].min(axis=1)
    tr = (df["High"] - df["Low"]).abs()
    noise = rng.lognormal(mean=0.0, sigma=0.35, size=len(df))
    base = (tr / tr.rolling(50).mean().bfill()).clip(lower=0.2).values
    df["Volume"] = (1e6 * base * noise).astype(int)
    return df[["Open", "High", "Low", "Close", "Volume"]]

# ---------- indicators ----------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def true_range(df):
    prev_close = df["Close"].shift(1)
    h_l = df["High"] - df["Low"]
    h_pc = (df["High"] - prev_close).abs()
    l_pc = (df["Low"] - prev_close).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

def atr(df, length=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

# ---------- signals ----------
def pa_signals(
    df,
    ema_fast_len=21,
    ema_trend_len=50,
    min_pullback_bars=2,
    fail_window=6,
    second_entry_window=12,
    vol_lookback=20,
    atr_lookback=20,
    vol_drop_mult=0.9,
    atr_drop_mult=0.9,
    allow_shorts=False
):
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["EMA21"] = ema(df["Close"], ema_fast_len)
    out["EMA50"] = ema(df["Close"], ema_trend_len)
    out["ATR"] = atr(df, 14)
    out["VOL_SMA"] = df["Volume"].rolling(vol_lookback).mean()
    out["ATR_SMA"] = out["ATR"].rolling(atr_lookback).mean()

    cross_up = (out["Close"] > out["EMA21"]) & (out["Close"].shift(1) <= out["EMA21"].shift(1))
    cross_dn = (out["Close"] < out["EMA21"]) & (out["Close"].shift(1) >= out["EMA21"].shift(1))
    uptrend   = out["Close"] > out["EMA50"]
    downtrend = out["Close"] < out["EMA50"]

    pullback_up = ((out["Close"] <= out["EMA21"]) & uptrend).astype(int)
    pullback_dn = ((out["Close"] >= out["EMA21"]) & downtrend).astype(int)
    pb_count_up = pullback_up.groupby((pullback_up != pullback_up.shift()).cumsum()).cumsum()
    pb_count_dn = pullback_dn.groupby((pullback_dn != pullback_dn.shift()).cumsum()).cumsum()

    vol_drop = df["Volume"] < (out["VOL_SMA"] * vol_drop_mult)
    atr_drop = out["ATR"] < (out["ATR_SMA"] * atr_drop_mult)
    drop_ok = vol_drop | atr_drop

    long_second = pd.Series(0, index=df.index, dtype=int)
    short_second = pd.Series(0, index=df.index, dtype=int)
    first_up_time = None
    first_up_high = None
    first_up_fail_time = None
    first_dn_time = None
    first_dn_low = None
    first_dn_fail_time = None

    for t in range(2, len(df)):
        i = df.index[t]
        if uptrend.iloc[t]:
            if pb_count_up.iloc[t-1] >= min_pullback_bars and cross_up.iloc[t]:
                if first_up_time is None or (i - first_up_time).days > second_entry_window:
                    first_up_time = i
                    first_up_high = df["High"].iloc[t]
                    first_up_fail_time = None
                else:
                    if first_up_fail_time is not None and (i - first_up_fail_time).days <= second_entry_window:
                        if df["High"].iloc[t] > first_up_high:
                            long_second.iloc[t] = 1
                        first_up_time = None
                        first_up_high = None
                        first_up_fail_time = None
            if first_up_time is not None:
                if cross_dn.iloc[t] and (i - first_up_time).days <= fail_window:
                    first_up_fail_time = i
        else:
            first_up_time = None
            first_up_high = None
            first_up_fail_time = None

        if allow_shorts and downtrend.iloc[t]:
            if pb_count_dn.iloc[t-1] >= min_pullback_bars and cross_dn.iloc[t]:
                if first_dn_time is None or (i - first_dn_time).days > second_entry_window:
                    first_dn_time = i
                    first_dn_low = df["Low"].iloc[t]
                    first_dn_fail_time = None
                else:
                    if first_dn_fail_time is not None and (i - first_dn_fail_time).days <= second_entry_window:
                        if df["Low"].iloc[t] < first_dn_low:
                            short_second.iloc[t] = 1
                        first_dn_time = None
                        first_dn_low = None
                        first_dn_fail_time = None
            if first_dn_time is not None:
                if cross_up.iloc[t] and (i - first_dn_time).days <= fail_window:
                    first_dn_fail_time = i
        else:
            first_dn_time = None
            first_dn_low = None
            first_dn_fail_time = None

    long_pullback = ((uptrend) &
                     (pb_count_up.shift(1) >= min_pullback_bars) &
                     cross_up &
                     drop_ok).astype(int)
    if allow_shorts:
        short_pullback = ((downtrend) &
                          (pb_count_dn.shift(1) >= min_pullback_bars) &
                          cross_dn &
                          drop_ok).astype(int)
    else:
        short_pullback = pd.Series(0, index=df.index, dtype=int)

    out["long_second_entry"] = long_second
    out["short_second_entry"] = short_second
    out["long_pullback_ema21"] = long_pullback
    out["short_pullback_ema21"] = short_pullback
    out["long_entry"]  = ((out["long_second_entry"] == 1) | (out["long_pullback_ema21"] == 1)).astype(int)
    out["short_entry"] = ((out["short_second_entry"] == 1) | (out["short_pullback_ema21"] == 1)).astype(int)
    out["long_exit"]  = ((out["Close"] < out["EMA21"]) | (out["Close"] < out["EMA50"])).astype(int)
    out["short_exit"] = ((out["Close"] > out["EMA21"]) | (out["Close"] > out["EMA50"])).astype(int)
    return out

# ---------- backtest ----------
def backtest(
    df_ohlc,
    signals_df,
    start_cash=10000.0,
    risk_per_trade=0.01,
    stop_atr_mult=2.0,
    tp_atr_mult=4.0,
    fee_bps=2.0,
    slippage_bps=1.0,
    allow_shorts=False
):
    fee = fee_bps * 1e-4
    slip = slippage_bps * 1e-4
    idx = df_ohlc.index
    close = df_ohlc["Close"]
    high = df_ohlc["High"]
    low = df_ohlc["Low"]
    atr_series = signals_df["ATR"]
    cash = start_cash
    position = None
    equity_curve = []
    trades = []
    for t in idx:
        c = close.loc[t]; h = high.loc[t]; l = low.loc[t]; atr_t = atr_series.loc[t]
        equity = cash if position is None else cash + position["qty"] * c * (1 if position["side"] == "long" else -1)
        if position is not None:
            side = position["side"]; qty = position["qty"]
            stop = position["stop"]; tp = position["tp"]
            entry_price = position["entry"]; entry_time = position["time"]
            exit_price = None; exit_reason = None
            if side == "long":
                if l <= stop: exit_price, exit_reason = stop * (1 - slip), "stop"
                elif h >= tp: exit_price, exit_reason = tp * (1 - slip), "tp"
                elif signals_df.loc[t, "long_exit"] == 1: exit_price, exit_reason = c * (1 - slip), "signal"
            else:
                if h >= stop: exit_price, exit_reason = stop * (1 + slip), "stop"
                elif l <= tp: exit_price, exit_reason = tp * (1 + slip), "tp"
                elif signals_df.loc[t, "short_exit"] == 1: exit_price, exit_reason = c * (1 + slip), "signal"
            if exit_price is not None:
                gross = qty * (exit_price - entry_price) * (1 if side == "long" else -1)
                fees = fee * (qty * entry_price + qty * exit_price)
                pnl = gross - fees
                cash += pnl
                trades.append({"entry_time": entry_time, "exit_time": t, "side": side,
                               "entry": entry_price, "exit": exit_price, "qty": qty,
                               "pnl": pnl, "reason": exit_reason})
                position = None
                equity = cash
        if position is None and not np.isnan(atr_t) and atr_t > 0:
            if signals_df.loc[t, "long_entry"] == 1:
                stop_dist = stop_atr_mult * atr_t
                qty = int(max((risk_per_trade * cash) // stop_dist, 0))
                if qty > 0:
                    entry_price = c * (1 + slip)
                    cash -= fee * qty * entry_price
                    position = {"side": "long", "qty": qty, "entry": entry_price,
                                "stop": entry_price - stop_dist, "tp": entry_price + tp_atr_mult * atr_t, "time": t}
            elif allow_shorts and signals_df.loc[t, "short_entry"] == 1:
                stop_dist = stop_atr_mult * atr_t
                qty = int(max((risk_per_trade * cash) // stop_dist, 0))
                if qty > 0:
                    entry_price = c * (1 - slip)
                    cash -= fee * qty * entry_price
                    position = {"side": "short", "qty": qty, "entry": entry_price,
                                "stop": entry_price + stop_dist, "tp": entry_price - tp_atr_mult * atr_t, "time": t}
        equity_curve.append(equity)
    eq = pd.Series(equity_curve, index=idx, name="Equity")
    trades_df = pd.DataFrame(trades)
    return eq, trades_df

# ---------- reporting ----------
def metrics(equity: pd.Series):
    ret = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0])**(1/years) - 1 if years > 0 else np.nan
    sharpe = np.sqrt(252) * ret.mean() / (ret.std() + 1e-12)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = dd.min()
    return {"Total Return": float(total_return), "CAGR": float(cagr), "Sharpe": float(sharpe), "Max Drawdown": float(mdd)}

def summarize_trades(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {"trades": 0}
    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] <= 0).sum()
    win_rate = wins / len(trades_df)
    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean()
    expectancy = trades_df["pnl"].mean()
    return {"trades": int(len(trades_df)), "win_rate": float(win_rate),
            "avg_win": float(avg_win) if not np.isnan(avg_win) else None,
            "avg_loss": float(avg_loss) if not np.isnan(avg_loss) else None,
            "expectancy": float(expectancy)}

def plot_results(df, equity):
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(df.index, df["Close"], label="Price")
    ax1.plot(df.index, ema(df["Close"], 21), label="EMA21", alpha=0.7)
    ax1.plot(df.index, ema(df["Close"], 50), label="EMA50", alpha=0.7)
    ax1.set_ylabel("Price")
    ax2 = ax1.twinx()
    ax2.plot(equity.index, equity.values, label="Equity", alpha=0.7)
    ax2.set_ylabel("Equity")
    fig.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
