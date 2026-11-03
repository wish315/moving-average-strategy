# -*- coding: utf-8 -*-
"""
回测引擎模块
包含：单次回测、参数网格搜索（含热力图）、滚动窗口回测主流程
交易成本：手续费万3（双边）+ 滑点万2（单边）
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product

from data_loader import load_index_data
from strategy import compute_signals
from metrics import (TRADING_DAYS, annualized_return, annualized_vol,
                     sharpe_ratio, max_drawdown, drawdown_series)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 全局配置（可修改） ----------------------
FEE_RATE = 0.0003               # 手续费 万3，双边
SLIPPAGE = 0.0002               # 滑点 单边万2
ROLL_TRAIN_WINDOW = 252 * 5     # 滚动训练窗口：5年训练
ROLL_VALID_WINDOW = 252 * 2     # 滚动验证窗口：2年样本外
START_SEARCH_DATE = "2006-01-01"
OUTPUT_DIR = "../results"

# 策略参数网格定义
PARAM_GRIDS = {
    "ma": {"short": [5, 8, 10, 15, 20], "long": [20, 30, 40, 60, 80]},
    "macd": {"fast": [8, 10, 12, 14], "slow": [20, 26, 30, 35]},
    "momentum": {"lookback": [10, 20, 30, 40, 60]},
    "donchian": {"n": [10, 20, 30, 40, 60]},
}
STRATEGY_NAMES = ["ma", "macd", "momentum", "donchian"]


# ===================== 单次回测 =====================
def run_backtest(df_raw: pd.DataFrame, strategy_type: str, p1, p2=None):
    """
    对给定区间执行一次策略回测

    Returns
    -------
    res : dict or None
        回测指标（含 df_valid 净值数据），样本不足时返回 None
    df : pd.DataFrame
        带信号的完整行情数据
    """
    df = compute_signals(df_raw, strategy_type, p1, p2)
    df["position"] = df["signal"].shift(1).fillna(0)
    trade_event = df["position"] != df["position"].shift(1)
    df["cost"] = 0.0
    df.loc[trade_event, "cost"] = FEE_RATE + SLIPPAGE

    df["daily_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["daily_return"] - df["cost"]
    df["benchmark_return"] = df["daily_return"]
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_bench"] = (1 + df["benchmark_return"]).cumprod()

    valid = df.dropna(subset=["cum_strategy"])
    if len(valid) < 20:
        return None, df

    ann_ret = annualized_return(valid["cum_strategy"])
    ann_bench = annualized_return(valid["cum_bench"])
    vol = annualized_vol(valid["strategy_return"])
    vol_bench = annualized_vol(valid["benchmark_return"])

    res = {
        "strategy": strategy_type,
        "p1": p1,
        "p2": p2,
        "total_ret": valid["cum_strategy"].iloc[-1] - 1,
        "ann_ret": ann_ret,
        "sharpe": sharpe_ratio(ann_ret, vol),
        "max_dd": max_drawdown(valid["cum_strategy"]),
        "bench_total_ret": valid["cum_bench"].iloc[-1] - 1,
        "bench_ann_ret": ann_bench,
        "bench_sharpe": sharpe_ratio(ann_bench, vol_bench),
        "bench_max_dd": max_drawdown(valid["cum_bench"]),
        "trade_cnt": int((df["signal"] != 0).sum()),
        "df_valid": valid,
    }
    return res, df


# ===================== 网格搜索 + 热力图 =====================
def grid_search_and_heatmap(df_train, strategy_name, param_grid, title_prefix):
    """单策略独立寻参，双参数策略输出夏普热力图"""
    results = []

    if strategy_name == "ma":
        for s, l in product(param_grid["short"], param_grid["long"]):
            if s >= l:
                continue
            ret, _ = run_backtest(df_train, "ma", p1=s, p2=l)
            if ret is None:
                continue
            results.append([s, l, ret["sharpe"], ret["ann_ret"], ret["max_dd"]])
        search_df = pd.DataFrame(results, columns=["short", "long", "sharpe", "ann_ret", "max_dd"])
        pivot = search_df.pivot(index="short", columns="long", values="sharpe")

    elif strategy_name == "macd":
        for f, s in product(param_grid["fast"], param_grid["slow"]):
            if f >= s:
                continue
            ret, _ = run_backtest(df_train, "macd", p1=f, p2=s)
            if ret is None:
                continue
            results.append([f, s, ret["sharpe"], ret["ann_ret"], ret["max_dd"]])
        search_df = pd.DataFrame(results, columns=["fast", "slow", "sharpe", "ann_ret", "max_dd"])
        pivot = search_df.pivot(index="fast", columns="slow", values="sharpe")

    elif strategy_name == "momentum":
        for look in param_grid["lookback"]:
            ret, _ = run_backtest(df_train, "momentum", p1=look)
            if ret is None:
                continue
            results.append([look, ret["sharpe"], ret["ann_ret"], ret["max_dd"]])
        search_df = pd.DataFrame(results, columns=["lookback", "sharpe", "ann_ret", "max_dd"])
        pivot = None

    elif strategy_name == "donchian":
        for n in param_grid["n"]:
            ret, _ = run_backtest(df_train, "donchian", p1=n)
            if ret is None:
                continue
            results.append([n, ret["sharpe"], ret["ann_ret"], ret["max_dd"]])
        search_df = pd.DataFrame(results, columns=["n", "sharpe", "ann_ret", "max_dd"])
        pivot = None

    else:
        raise ValueError(f"未知策略类型：{strategy_name}")

    if pivot is not None:
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
        plt.title(f"{title_prefix} | {strategy_name} 夏普热力图")
        plt.tight_layout()
        plt.show()

    best_row = search_df.loc[search_df["sharpe"].idxmax()]
    return best_row, search_df


# ===================== 滚动窗口回测主流程 =====================
def rolling_backtest(df_raw):
    """5年训练 + 2年样本外验证的滚动窗口回测"""
    start_idx = ROLL_TRAIN_WINDOW
    perf_records = []
    all_cum_data = {}
    all_dd_data = {}

    while start_idx + ROLL_VALID_WINDOW <= len(df_raw):
        train_df = df_raw.iloc[start_idx - ROLL_TRAIN_WINDOW: start_idx].copy()
        valid_df = df_raw.iloc[start_idx: start_idx + ROLL_VALID_WINDOW].copy()
        roll_start = valid_df.index.min().date()
        roll_end = valid_df.index.max().date()
        print(f"\n===== 滚动窗口：验证区间 {roll_start} ~ {roll_end} =====")

        for st_name in STRATEGY_NAMES:
            best_param, _ = grid_search_and_heatmap(
                train_df, st_name, PARAM_GRIDS[st_name], f"窗口{roll_start}")

            if st_name == "ma":
                p1, p2 = int(best_param["short"]), int(best_param["long"])
            elif st_name == "macd":
                p1, p2 = int(best_param["fast"]), int(best_param["slow"])
            elif st_name == "momentum":
                p1, p2 = int(best_param["lookback"]), None
            else:  # donchian
                p1, p2 = int(best_param["n"]), None

            res, _ = run_backtest(valid_df, st_name, p1, p2)
            if res is None:
                continue
            print(f"【{st_name}】最优参数 p1={p1},p2={p2} | "
                  f"夏普={res['sharpe']:.2f} | 年化={res['ann_ret']:.2%} | "
                  f"最大回撤={res['max_dd']:.2%}")

            perf_records.append({
                "roll_start": roll_start, "roll_end": roll_end,
                "strategy": st_name, "p1": p1, "p2": p2,
                "ann_ret": res["ann_ret"], "sharpe": res["sharpe"],
                "max_dd": res["max_dd"], "total_ret": res["total_ret"],
                "trade_cnt": res["trade_cnt"],
                "bench_ann_ret": res["bench_ann_ret"],
                "bench_sharpe": res["bench_sharpe"],
                "bench_max_dd": res["bench_max_dd"],
            })

            key = f"{st_name}_{roll_start}"
            all_cum_data[key] = res["df_valid"][["cum_strategy", "cum_bench"]].copy()
            all_dd_data[key] = drawdown_series(res["df_valid"]["cum_strategy"])

        start_idx += ROLL_VALID_WINDOW

    return pd.DataFrame(perf_records), all_cum_data, all_dd_data


# ===================== 绘图 =====================
def plot_equity_and_drawdown(all_cum_data, all_dd_data):
    """上子图净值曲线，下子图回撤序列"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for k, cum in all_cum_data.items():
        s_name = k.split("_")[0]
        ax1.plot(cum.index, cum["cum_strategy"], label=f"{s_name}策略", alpha=0.7)
        ax1.plot(cum.index, cum["cum_bench"], c="k", ls="--", alpha=0.4, label="买入持有")
    ax1.set_title("滚动窗口回测：各策略净值曲线")
    ax1.set_ylabel("净值")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for k, dd in all_dd_data.items():
        ax2.plot(dd.index, dd, label=f"{k.split('_')[0]}回撤", alpha=0.7)
    ax2.set_title("策略最大回撤序列")
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_yearly_returns(perf_df):
    """各策略年度平均年化收益柱状图"""
    perf_df = perf_df.copy()
    perf_df["year"] = pd.to_datetime(perf_df["roll_start"]).dt.year
    year_ret = perf_df.groupby(["year", "strategy"])["ann_ret"].mean().unstack()
    year_ret.plot(kind="bar", figsize=(12, 6))
    plt.title("各策略年度平均年化收益")
    plt.ylabel("年化收益")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ===================== 主入口 =====================
if __name__ == "__main__":
    # 1. 加载数据
    df_raw = load_index_data(start_date=START_SEARCH_DATE)
    print(f"数据区间：{df_raw.index.min().date()} ~ {df_raw.index.max().date()}，共 {len(df_raw)} 个交易日")

    # 2. 滚动窗口回测
    perf_df, all_cum_data, all_dd_data = rolling_backtest(df_raw)

    # 3. 绩效汇总表导出
    perf_df.to_csv(f"{OUTPUT_DIR}/backtest_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 绩效表已保存：{OUTPUT_DIR}/backtest_summary.csv")
    print(perf_df.to_string(index=False))

    # 4. 可视化
    plot_equity_and_drawdown(all_cum_data, all_dd_data)
    plot_yearly_returns(perf_df)
