# -*- coding: utf-8 -*-
"""
数据获取模块
基于 akshare 获取指数日行情数据，并整理为标准 OHLCV 格式

健壮性设计：
1. 多数据源自动回退：index_zh_a_hist(东财) -> stock_zh_index_daily_em(东财)
   -> stock_zh_index_daily(新浪)，任一接口失效自动切换
2. 列名归一化：兼容中文（日期/开盘/收盘...）与英文（date/open/...）列名
3. 空结果检测：返回空 DataFrame 时给出明确提示（常见原因是 akshare 版本过旧）
"""

import akshare as ak
import pandas as pd

# 目标列名 -> 可能的原始列名（兼容中英文、大小写）
COLUMN_ALIASES = {
    "Date": ["date", "Date", "日期"],
    "Open": ["open", "Open", "开盘"],
    "High": ["high", "High", "最高"],
    "Low": ["low", "Low", "最低"],
    "Close": ["close", "Close", "收盘"],
    "Volume": ["volume", "Volume", "成交量"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将任意命名的 OHLCV 列映射为标准列名"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]  # 去空白
    rename_map = {}
    for std_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = std_name
                break  # 命中一个即可
    df.rename(columns=rename_map, inplace=True)

    missing = [c for c in COLUMN_ALIASES if c not in df.columns]
    if missing:
        raise KeyError(
            f"数据缺少必要列 {missing}。实际列名：{list(df.columns)}。"
            f"请检查 akshare 接口版本或 symbol 参数。"
        )
    return df[list(COLUMN_ALIASES)]


def _try_fetch(fetch_fn, source_name: str) -> pd.DataFrame | None:
    """尝试单个数据源；空结果 / 报错时返回 None 并打印提示"""
    try:
        df = fetch_fn()
        if df is None or len(df) == 0 or len(df.columns) == 0:
            print(f"[数据源] {source_name} 返回空数据，尝试下一个数据源...")
            return None
        print(f"[数据源] 使用 {source_name}，获取 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"[数据源] {source_name} 调用失败（{type(e).__name__}: {e}），尝试下一个数据源...")
        return None


def load_index_data(symbol: str = "000300",
                    start_date: str = "2006-01-01") -> pd.DataFrame:
    """
    获取指数日行情数据，多数据源自动回退

    Parameters
    ----------
    symbol : str
        指数代码，默认 "000300"（沪深300）
    start_date : str
        数据起始日期，格式 "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame
        以 Date 为索引的行情数据，包含 Open / High / Low / Close / Volume
    """
    start_compact = start_date.replace("-", "")
    end_compact = "20991231"

    # 数据源1：东财指数历史行情（返回中文列名，支持起止日期）
    df = _try_fetch(
        lambda: ak.index_zh_a_hist(symbol=symbol, period="daily",
                                   start_date=start_compact, end_date=end_compact),
        "index_zh_a_hist (东方财富)")

    # 数据源2：东财指数日行情（返回英文小写列名）
    if df is None:
        df = _try_fetch(
            lambda: ak.stock_zh_index_daily_em(symbol=symbol),
            "stock_zh_index_daily_em (东方财富)")

    # 数据源3：新浪指数日行情（symbol 需带交易所前缀）
    if df is None:
        sina_symbol = ("sh" + symbol) if symbol.startswith(("0", "9")) else ("sz" + symbol)
        df = _try_fetch(
            lambda: ak.stock_zh_index_daily(symbol=sina_symbol),
            "stock_zh_index_daily (新浪)")

    if df is None:
        raise RuntimeError(
            "所有数据源均获取数据失败。请排查：\n"
            "  1. 网络是否能访问东方财富/新浪财经；\n"
            "  2. akshare 版本是否过旧，建议升级：pip install -U akshare"
        )

    df = _normalize_columns(df)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.set_index("Date", inplace=True)
    df = df[df.index >= start_date].copy()

    if len(df) == 0:
        raise RuntimeError(f"过滤 {start_date} 之后无数据，请检查 start_date 参数。")

    return df


if __name__ == "__main__":
    df = load_index_data()
    print(f"数据区间：{df.index.min().date()} ~ {df.index.max().date()}，共 {len(df)} 个交易日")
    print(df.tail())
