# -*- coding: utf-8 -*-
"""
策略信号模块
实现 4 种趋势跟踪策略的信号生成，统一叠加 MA250 年线趋势过滤：
  1. ma       ：短长期均线金叉/死叉
  2. macd     ：MACD (DIFF/DEA) 金叉/死叉
  3. momentum ：动量翻正/翻负
  4. donchian ：唐奇安通道突破/跌破
"""

import pandas as pd

TREND_MA = 250  # 趋势过滤均线：250日年线


def compute_signals(df: pd.DataFrame, strategy_type: str,
                    p1, p2=None) -> pd.DataFrame:
    """
    在行情 DataFrame 上计算策略信号

    Parameters
    ----------
    df : pd.DataFrame
        行情数据（Open/High/Low/Close），原始索引保留
    strategy_type : str
        策略类型：ma / macd / momentum / donchian
    p1, p2 : 策略参数

    Returns
    -------
    pd.DataFrame
        增加 signal（1=买入，-1=卖出，0=持有原仓位）的 DataFrame
    """
    df = df.copy()
    df["trend_ma"] = df["Close"].rolling(TREND_MA).mean()
    trend_ok = df["Close"] > df["trend_ma"]
    df["signal"] = 0

    if strategy_type == "ma":
        s, l = p1, p2
        df["ma_s"] = df["Close"].rolling(s).mean()
        df["ma_l"] = df["Close"].rolling(l).mean()
        buy = (df["ma_s"] > df["ma_l"]) & (df["ma_s"].shift(1) <= df["ma_l"].shift(1)) & trend_ok
        sell = (df["ma_s"] < df["ma_l"]) & (df["ma_s"].shift(1) >= df["ma_l"].shift(1))

    elif strategy_type == "macd":
        fast, slow = p1, p2
        df["ema_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
        df["diff"] = df["ema_fast"] - df["ema_slow"]
        df["dea"] = df["diff"].ewm(span=9, adjust=False).mean()
        buy = (df["diff"] > df["dea"]) & (df["diff"].shift(1) <= df["dea"].shift(1)) & trend_ok
        sell = (df["diff"] < df["dea"]) & (df["diff"].shift(1) >= df["dea"].shift(1))

    elif strategy_type == "momentum":
        lookback = p1
        df["mom"] = df["Close"] / df["Close"].shift(lookback) - 1
        buy = (df["mom"] > 0) & (df["mom"].shift(1) <= 0) & trend_ok
        sell = (df["mom"] < 0) & (df["mom"].shift(1) >= 0)

    elif strategy_type == "donchian":
        n = p1
        df["donchian_high"] = df["High"].rolling(n).max()
        df["donchian_low"] = df["Low"].rolling(n).min()
        buy = (df["Close"] > df["donchian_high"].shift(1)) & trend_ok
        sell = (df["Close"] < df["donchian_low"].shift(1))

    else:
        raise ValueError(f"未知策略类型：{strategy_type}")

    df.loc[buy, "signal"] = 1
    df.loc[sell, "signal"] = -1
    return df
