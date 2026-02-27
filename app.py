import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cross-Qualifying Ticker Finder", layout="wide")

# -----------------------------
# Constants
# -----------------------------
META_COLS_DEFAULT = ["Symbol", "Exchange", "Company", "Sector", "Rank"]
KNOWN_SUFFIXES = (".TO", ".V", ".CN", ".NE", ".TSX", ".TSXV")

# -----------------------------
# Normalization
# -----------------------------
def normalize_ticker(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    if not s:
        return ""

    if ":" in s:
        s = s.split(":")[-1].strip()

    for suf in KNOWN_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break

    s = re.sub(r"[,\s]+", "", s)
    return s


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_table_from_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_base(base_path: str, uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return load_table_from_upload(uploaded_file)

    p = Path(base_path)
    if not p.exists():
        raise FileNotFoundError(f"Base file not found: {base_path}")

    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)


def build_symbol_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    symbol_col = "Symbol" if "Symbol" in out.columns else out.columns[0]
    out["__ticker_key__"] = out[symbol_col].astype(str).apply(normalize_ticker)
    out["__symbol_key__"] = out[symbol_col].astype(str).str.strip().str.upper()
    return out


def infer_meta_cols(df: pd.DataFrame) -> List[str]:
    meta = [c for c in META_COLS_DEFAULT if c in df.columns]
    if "Symbol" in df.columns and "Symbol" not in meta:
        meta = ["Symbol"] + meta
    return meta


def get_qualifier_cols(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    ignore = set(meta_cols + ["__ticker_key__", "__symbol_key__"])
    return [c for c in df.columns if c not in ignore]


# -----------------------------
# Parsing Inputs
# -----------------------------
def parse_manual_tickers(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\n,; \t]+", text)
    keys = [normalize_ticker(p) for p in parts if p]
    return list(dict.fromkeys([k for k in keys if k]))


def parse_uploaded_ticker_file(uploaded) -> List[str]:
    if uploaded is None:
        return []
    df = load_table_from_upload(uploaded)
    if df.empty:
        return []
    col = df.columns[0]
    keys = df[col].dropna().astype(str).apply(normalize_ticker)
    return list(dict.fromkeys([k for k in keys if k]))


# -----------------------------
# Matching Engine
# -----------------------------
def find_memberships(
    df_indexed: pd.DataFrame,
    ticker_keys: List[str],
    meta_cols: List[str],
    qualifier_cols: List[str],
    match_mode: str = "Ticker",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    long_rows = []
    matrix_rows: Dict[str, Dict[str, float]] = {}

    key_col = "__symbol_key__" if match_mode == "Exact Symbol" else "__ticker_key__"

    for input_key in ticker_keys:
        matches = df_indexed[df_indexed[key_col] == input_key]

        if matches.empty:
            long_rows.append({
                "Input": input_key,
                "Found?": False,
                "Matched Symbol": "",
                "Membership Columns": "",
                "Membership Count": 0,
            })
            continue

        matrix_rows.setdefault(input_key, {})

        for _, r in matches.iterrows():
            memberships = []
            for c in qualifier_cols:
                v = r.get(c)
                if pd.notna(v) and str(v).strip() != "":
                    memberships.append(c)
                    matrix_rows[input_key][c] = 1

            row = {
                "Input": input_key,
                "Found?": True,
                "Matched Symbol": r.get("Symbol", ""),
                "Membership Columns": ", ".join(memberships),
                "Membership Count": len(memberships),
            }

            for mc in meta_cols:
                if mc in matches.columns:
                    row[mc] = r.get(mc, "")

            long_rows.append(row)

    long_results = pd.DataFrame(long_rows)
    matrix = pd.DataFrame.from_dict(matrix_rows, orient="index").fillna(0)
    matrix.index.name = "Input"

    return long_results, matrix


# -----------------------------
# Portfolio Helpers
# -----------------------------
def clean_weight_series(s: pd.Series) -> pd.Series:
    s2 = (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    s2 = pd.to_numeric(s2, errors="coerce")

    if not s2.dropna().empty:
        if (s2.dropna() <= 1).mean() > 0.7:
            s2 = s2 * 100.0

    return s2.fillna(0.0)


# -----------------------------
# Charts (SAFE)
# -----------------------------
def chart_pie(values: List[float], labels: List[str], title: str):
    cleaned = [0 if (v is None or pd.isna(v) or not np.isfinite(v)) else float(v) for v in values]
    total = sum(cleaned)

    if total <= 0:
        st.info(f"{title}: No valid data to display.")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.pie(cleaned, labels=labels,
           autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
           startangle=90)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


def chart_hist(series: pd.Series, title: str):
    values = series.dropna().astype(float)
    if values.empty:
        st.info(f"{title}: No data available.")
        return

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(values, bins=10)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# UI
# -----------------------------
st.title("Cross-Qualifying Ticker Finder")

with st.sidebar:
    base_upload = st.file_uploader("Upload base file", type=["csv", "xlsx", "xls"])
    base_path = st.text_input("Base path", "Cross Qualifying Stocks.csv")
    mode = st.radio("Mode", ["Ticker list", "Portfolio (with weights)"])
    match_mode = st.radio("Match using", ["Ticker", "Exact Symbol"])

# Load base
base_df = load_base(base_path, base_upload)
base_df = build_symbol_index(base_df)
meta_cols = infer_meta_cols(base_df)
qualifier_cols = get_qualifier_cols(base_df, meta_cols)

ticker_keys = []
portfolio_df = None

if mode == "Ticker list":
    manual = st.text_area("Paste tickers")
    upload = st.file_uploader("Upload ticker list")
    ticker_keys = parse_manual_tickers(manual) + parse_uploaded_ticker_file(upload)
else:
    upload = st.file_uploader("Upload portfolio")
    if upload:
        portfolio_df = load_table_from_upload(upload)
        ticker_col = portfolio_df.columns[0]
        weight_col = portfolio_df.columns[1]

        portfolio_df["__ticker_key__"] = portfolio_df[ticker_col].apply(normalize_ticker)
        portfolio_df["__weight_pct__"] = clean_weight_series(portfolio_df[weight_col])
        ticker_keys = portfolio_df["__ticker_key__"].tolist()

ticker_keys = list(dict.fromkeys([t for t in ticker_keys if t]))

if not ticker_keys:
    st.stop()

long_results, matrix = find_memberships(
    base_df, ticker_keys, meta_cols, qualifier_cols, match_mode
)

st.dataframe(long_results, use_container_width=True)

# -----------------------------
# Portfolio Insights
# -----------------------------
if mode == "Portfolio (with weights)" and portfolio_df is not None:

    weights = (
        portfolio_df.groupby("__ticker_key__", as_index=False)["__weight_pct__"]
        .sum()
        .rename(columns={"__ticker_key__": "Input", "__weight_pct__": "Weight (%)"})
    )

    mapped = weights["Input"].isin(matrix.index)
    mapped_w = float(weights.loc[mapped, "Weight (%)"].sum())
    unmapped_w = float(weights.loc[~mapped, "Weight (%)"].sum())
    total_w = float(weights["Weight (%)"].sum())

    st.metric("Total Weight (%)", f"{total_w:.2f}")
    st.metric("Mapped Weight (%)", f"{mapped_w:.2f}")
    st.metric("Unmapped Weight (%)", f"{unmapped_w:.2f}")

    if mapped_w > 0 or unmapped_w > 0:
        chart_pie([mapped_w, unmapped_w], ["Mapped", "Unmapped"], "Coverage by weight")

    chart_hist(long_results["Membership Count"], "Membership Distribution")

# -----------------------------
# Download
# -----------------------------
st.download_button(
    "Download CSV",
    long_results.to_csv(index=False).encode("utf-8"),
    "results.csv",
    "text/csv",
)
