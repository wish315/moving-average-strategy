# -*- coding: utf-8 -*-
"""
绩效指标模块
计算年化收益、夏普比率、最大回撤等核心回测指标
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # 年化交易日
RISK_FREE = 0.02    # 年化无风险利率


def annualized_return(cum_series: pd.Series) -> float:
    """由净值序列计算年化收益率"""
    total_ret = cum_series.iloc[-1] - 1
    n = len(cum_series)
    return (1 + total_ret) ** (TRADING_DAYS / n) - 1


def annualized_vol(ret_series: pd.Series) -> float:
    """由日收益序列计算年化波动率"""
    return ret_series.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(ann_ret: float, ann_vol: float) -> float:
    """夏普比率 = (年化收益 - 无风险利率) / 年化波动率"""
    if ann_vol > 1e-6:
        return (ann_ret - RISK_FREE) / ann_vol
    return -999.0


def max_drawdown(cum_series: pd.Series) -> float:
    """最大回撤（负值）"""
    roll_max = cum_series.cummax()
    dd = cum_series / roll_max - 1
    return dd.min()


def drawdown_series(cum_series: pd.Series) -> pd.Series:
    """回撤序列（用于绘图）"""
    return cum_series / cum_series.cummax() - 1
