import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "/Users/seunghoonlee/Downloads/Altman_paper/edgar"  # 너 폴더
QUARTERS = ["2021q1", "2021q2", "2021q3", "2021q4"]
FY_TARGET = 2020

# EDGAR tags we need
STOCK_TAGS = ["Assets", "AssetsCurrent", "LiabilitiesCurrent", "Liabilities"]  # balance sheet (point-in-time)
FLOW_TAGS  = ["OperatingIncomeLoss", "Revenues"]                                # income statement (period)

# retained earnings candidates (you already validated this approach)
RE_CANDIDATES = [
    "RetainedEarningsAccumulatedDeficit",
    "RetainedEarnings",
    "RetainedEarningsUnappropriated",
    "AccumulatedDeficit",
    "ShareholdersRetainedEarningsAccumulatedDeficit",
    "ManagerRetainedEarningsAccumulatedDeficit",
]

# Shares proxy for X4
SHARES_TAG = "WeightedAverageNumberOfDilutedSharesOutstanding"

# -----------------------------
# HELPERS
# -----------------------------
def read_sub_num(quarter_dir: str):
    """Read sub.txt and num.txt for one quarter."""
    sub_path = os.path.join(quarter_dir, "sub.txt")
    num_path = os.path.join(quarter_dir, "num.txt")

    # SEC "Financial Statement Data Sets" are tab-delimited
    sub = pd.read_csv(sub_path, sep="\t", low_memory=False)
    num = pd.read_csv(num_path, sep="\t", low_memory=False)

    return sub, num

def pick_absmax_per_adsh(df: pd.DataFrame, value_col="value"):
    """
    For each adsh, pick the row with largest |value|.
    Safe against NaNs / non-numeric values.
    """
    if df is None or df.empty:
        return df

    d = df.copy()

    # make numeric and drop NaNs
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])

    if d.empty:
        return d

    # idxmax per adsh on abs(value)
    idx = d[value_col].abs().groupby(d["adsh"]).idxmax()

    # drop NaN indices just in case
    idx = idx.dropna()

    if len(idx) == 0:
        return d.iloc[0:0].copy()

    return d.loc[idx].copy()

def ensure_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def get_cik_to_ticker_map():
    """
    Get CIK -> ticker mapping from SEC company_tickers.json.
    (Requires internet.)
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "AltmanZScoreResearch/1.0 (contact: youremail@example.com)"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    rows = []
    # file format: {"0": {"cik_str": 320213, "ticker": "AAPL", ...}, ...}
    for _, v in data.items():
        rows.append((int(v["cik_str"]), v["ticker"]))
    m = pd.DataFrame(rows, columns=["cik", "ticker"]).drop_duplicates("cik")
    return m

import time
import numpy as np
import pandas as pd
import yfinance as yf

def get_year_end_close(tickers, start="2020-12-01", end="2021-01-10", sleep_sec=0.2):
    tickers = [t for t in tickers if isinstance(t, str) and t]
    tickers = sorted(set(tickers))

    closes = {}
    for i, t in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"priced {i}/{len(tickers)}")
        try:
            hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=False)
            if hist is None or hist.empty or ("Close" not in hist.columns):
                closes[t] = np.nan
            else:
                target = pd.Timestamp("2020-12-31")

                # yfinance index가 tz-aware일 때가 있어서 안전하게 tz 제거
                h = hist.copy()
                try:
                    h.index = h.index.tz_localize(None)
                except Exception:
                    pass

                s = h.loc[:target, "Close"].dropna()   # target 날짜 "이전까지" 중 마지막 거래일
                closes[t] = float(s.iloc[-1]) if len(s) else np.nan
        except Exception:
            closes[t] = np.nan
        time.sleep(sleep_sec)

    return pd.Series(closes, name="Close_2020YE")

def get_year_end_close_safe(tickers, start="2020-12-01", end="2021-01-10", sleep_sec=0.2):
    """
    Robust: fetch per-ticker using Ticker().history() so we always get simple columns.
    Returns Series: ticker -> last raw Close in window.
    """
    tickers = [t for t in tickers if isinstance(t, str) and len(t) > 0]
    tickers = sorted(set(tickers))
    print(">>> USING FIXED get_year_end_close_safe <<<")
    closes = {}
    for i, t in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"priced {i}/{len(tickers)}")
        try:
            hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=False)
            if hist is None or hist.empty or ("Close" not in hist.columns):
                closes[t] = np.nan
            else:
                h = hist.copy()
                try:
                    h.index = h.index.tz_localize(None)   # tz 제거 (안 되면 그냥 넘어감)
                except Exception:
                    pass

                target = pd.Timestamp("2020-12-31")
                s = h.loc[:target, "Close"].dropna()      # 2020-12-31 이전까지의 마지막 종가
                closes[t] = float(s.iloc[-1]) if len(s) else np.nan
        except Exception as e:
            closes[t] = np.nan
        time.sleep(sleep_sec)  # rate-limit 방지

    return pd.Series(closes, name="Close_2020YE")


# -----------------------------
# 1) LOAD + CONCAT 2021q1~q4
# -----------------------------
subs = []
nums = []

for q in QUARTERS:
    qdir = os.path.join(BASE_DIR, q)
    print("loading", qdir)
    sub_q, num_q = read_sub_num(qdir)
    subs.append(sub_q)
    nums.append(num_q)

sub_all = pd.concat(subs, ignore_index=True)
num_all = pd.concat(nums, ignore_index=True)

# Make sure key columns are numeric where needed
sub_all = ensure_numeric(sub_all, ["fy", "cik", "period"])
num_all = ensure_numeric(num_all, ["ddate", "qtrs", "value"])

print("sub_all:", sub_all.shape, "num_all:", num_all.shape)

# -----------------------------
# 2) FILTER: FY=2020, Form=10-K
# -----------------------------
# Keep only 10-K for FY_TARGET
sub_2020 = sub_all[(sub_all["form"] == "10-K") & (sub_all["fy"] == FY_TARGET)].copy()

# adsh -> (period, cik, name)
adsh_meta = sub_2020[["adsh", "period", "cik", "name"]].drop_duplicates("adsh")
print("2020 10-K filings:", adsh_meta.shape)

# Limit num to only those adsh to reduce work
num_2020 = num_all.merge(adsh_meta[["adsh", "period"]], on="adsh", how="inner")

# Critical: keep only facts at the fiscal-year-end date
num_2020 = num_2020[num_2020["ddate"] == num_2020["period"]].copy()

print("num_2020 (after adsh+ddate==period):", num_2020.shape)

# -----------------------------
# 3) EXTRACT VARIABLES
#    - Stock items: qtrs == 0
#    - Flow items: prefer qtrs == 4 (annual), fallback: if missing, allow qtrs != 0 later
# -----------------------------
# ---- Stock: Assets, AssetsCurrent, LiabilitiesCurrent, Liabilities
stock = num_2020[(num_2020["tag"].isin(STOCK_TAGS)) & (num_2020["qtrs"] == 0)].copy()

# Pick abs-max per adsh+tag then pivot
stock = (stock
         .sort_values("value", key=lambda x: x.abs(), ascending=False)
         .drop_duplicates(["adsh", "tag"])
        )

stock_p = stock.pivot(index="adsh", columns="tag", values="value")

# ---- Retained earnings (X2 numerator): qtrs==0, ddate==period already enforced
re_df = num_2020[(num_2020["tag"].isin(RE_CANDIDATES)) & (num_2020["qtrs"] == 0)].copy()
re_df["value"] = pd.to_numeric(re_df["value"], errors="coerce")

# pick one retained earnings number per adsh by abs max
re_one = pick_absmax_per_adsh(re_df)
re_one = re_one[["adsh", "value"]].rename(columns={"value": "RetainedEarnings_proxy"})

# ---- Flow: Revenues, OperatingIncomeLoss
flow = num_2020[(num_2020["tag"].isin(FLOW_TAGS + [SHARES_TAG]))].copy()

# Prefer annual qtrs==4 for flow + shares; if a company reports differently, we can fallback later
flow4 = flow[flow["qtrs"] == 4].copy()

# For each tag, keep abs-max per adsh+tag (consolidated)
flow4 = (flow4
         .sort_values("value", key=lambda x: x.abs(), ascending=False)
         .drop_duplicates(["adsh", "tag"])
        )

flow_p = flow4.pivot(index="adsh", columns="tag", values="value")

# -----------------------------
# 4) BUILD MASTER TABLE
# -----------------------------
df = adsh_meta.set_index("adsh").join(stock_p, how="left").join(re_one.set_index("adsh"), how="left").join(flow_p, how="left")

# Rename for clarity
df = df.rename(columns={
    "Assets": "A",
    "AssetsCurrent": "CA",
    "LiabilitiesCurrent": "CL",
    "Liabilities": "TL",
    "Revenues": "S",
    "OperatingIncomeLoss": "EBIT",
    SHARES_TAG: "SHARES_DIL_WAVG",
})

# Compute working capital
df["WC"] = df["CA"] - df["CL"]

# -----------------------------
# 5) COMPUTE Z_noMVE (X1,X2,X3,X5)
# -----------------------------
# Keep only rows with required accounting fields
req_cols = ["A", "CA", "CL", "TL", "S", "EBIT", "RetainedEarnings_proxy"]
df["Z_noMVE"] = np.nan

mask = df[["A", "CA", "CL", "S", "EBIT", "RetainedEarnings_proxy"]].notna().all(axis=1) & (df["A"] != 0)

df.loc[mask, "Z_noMVE"] = (
    1.2 * (df.loc[mask, "WC"] / df.loc[mask, "A"]) +
    1.4 * (df.loc[mask, "RetainedEarnings_proxy"] / df.loc[mask, "A"]) +
    3.3 * (df.loc[mask, "EBIT"] / df.loc[mask, "A"]) +
    1.0 * (df.loc[mask, "S"] / df.loc[mask, "A"])
)

print("Z_noMVE non-null:", df["Z_noMVE"].notna().sum())

# -----------------------------
# 6) ADD X4 via PRICE (Yahoo) * SHARES (EDGAR proxy) / TL
#    Need cik -> ticker mapping, then year-end raw close
# -----------------------------
# Get tickers from SEC mapping (internet required)
cik_ticker = get_cik_to_ticker_map()

df2 = df.reset_index().merge(cik_ticker, on="cik", how="left")

# Download year-end close for all tickers that exist
tickers = df2["ticker"].dropna().unique().tolist()
print("Tickers to price:", len(tickers))

close_map = get_year_end_close_safe(tickers, start="2020-12-01", end="2021-01-10", sleep_sec=0.2)
close_df = close_map.rename_axis("ticker").reset_index()
df2 = df2.merge(close_df, on="ticker", how="left")

# Compute X4 and Z_full (only if we have Z_noMVE + Close + Shares + TL)
df2["X4"] = np.nan
mask_x4 = df2["Close_2020YE"].notna() & df2["SHARES_DIL_WAVG"].notna() & df2["TL"].notna() & (df2["TL"] != 0)
df2.loc[mask_x4, "X4"] = (df2.loc[mask_x4, "Close_2020YE"] * df2.loc[mask_x4, "SHARES_DIL_WAVG"]) / df2.loc[mask_x4, "TL"]

df2["Z_full"] = np.nan
mask_full = df2["Z_noMVE"].notna() & df2["X4"].notna()
df2.loc[mask_full, "Z_full"] = df2.loc[mask_full, "Z_noMVE"] + 0.6 * df2.loc[mask_full, "X4"]

print("Z_full non-null:", df2["Z_full"].notna().sum())

# -----------------------------
# 7) OUTPUT
# -----------------------------
out_cols = [
    "adsh", "cik", "ticker", "name", "period",
    "A", "CA", "CL", "TL", "WC",
    "RetainedEarnings_proxy", "EBIT", "S",
    "SHARES_DIL_WAVG", "Close_2020YE",
    "Z_noMVE", "X4", "Z_full",
]
out = df2[out_cols].copy()

out_path = os.path.join(BASE_DIR, "altman_2020_all_companies.csv")
out.to_csv(out_path, index=False)
print("Saved:", out_path)

# Optional quick sanity checks
print(out["Z_full"].describe())
print("Example GM rows:")
print(out[out["ticker"]=="GM"][["name","Z_noMVE","X4","Z_full"]].head())
