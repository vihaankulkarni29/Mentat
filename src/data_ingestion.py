"""Data ingestion and feature engineering for Mentat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-not-found]


def fetch_stock_data(ticker: str, lookback_years: int = 3) -> pd.DataFrame:
    """Fetch adjusted OHLCV data for a ticker."""
    period = f"{lookback_years}y"
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False, timeout=15)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    return df


def fetch_vix(lookback_years: int = 3) -> pd.Series:
    """Fetch VIX as macro volatility proxy."""
    period = f"{lookback_years}y"
    vix = yf.download("^VIX", period=period, auto_adjust=True, progress=False, timeout=15)
    if vix is None or vix.empty:
        return pd.Series(dtype=float, name="VIX")

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    return vix["Close"].rename("VIX").dropna()


def fetch_market_returns(benchmark_ticker: str, lookback_years: int = 3) -> pd.Series:
    """Fetch benchmark returns for beta calculation."""
    period = f"{lookback_years}y"
    idx = yf.download(benchmark_ticker, period=period, auto_adjust=True, progress=False, timeout=15)
    if idx is None or idx.empty:
        return pd.Series(dtype=float, name="market_return")

    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.get_level_values(0)

    return idx["Close"].pct_change().rename("market_return").dropna()


def build_feature_matrix(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Build observation matrix from OHLCV + macro for HMM."""
    feat = pd.DataFrame(index=df.index)

    # Layer 2 core features
    feat["log_ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    feat["log_ret_5d"] = np.log(df["Close"] / df["Close"].shift(5))
    feat["log_ret_20d"] = np.log(df["Close"] / df["Close"].shift(20))

    feat["rvol_10d"] = feat["log_ret_1d"].rolling(10).std() * np.sqrt(252)
    feat["rvol_20d"] = feat["log_ret_1d"].rolling(20).std() * np.sqrt(252)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi"] = 100 - (100 / (1 + rs))

    vol_mean_20 = df["Volume"].rolling(20).mean()
    vol_std_20 = df["Volume"].rolling(20).std()
    feat["vol_zscore"] = (df["Volume"] - vol_mean_20) / vol_std_20

    feat["vix"] = vix.reindex(df.index, method="ffill")

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return feat
