import re
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cross-Qualifying Ticker Finder", layout="wide")

# ============================================================
# CONSTANTS
# ============================================================

META_COLS_DEFAULT = ["Symbol", "Exchange", "Company", "Sector", "Rank"]
KNOWN_SUFFIXES = (".TO", ".V", ".CN", ".NE", ".TSX", ".TSXV")

# ============================================================
# HELPERS
# ============================================================

def normalize_ticker(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip().upper()

    if ":" in s:
        s = s.split(":")[-1]

    for suf in KNOWN_SUFFIXES:
        if s.endswith(suf):
            s = s[:-len(suf)]
            break

    s = re.sub(r"[,\s]+", "", s)
    return s


@st.cache_data(show_spinner=False)
def load_table(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(("xlsx", "xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_base(base_path: str, uploaded=None):
    if uploaded is not None:
        return load_table(uploaded)

    p = Path(base_path)
    if not p.exists():
        st.error(f"Base file not found at: {base_path}")
        st.stop()

    if p.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)


def infer_meta_cols(df: pd.DataFrame) -> List[str]:
    meta = [c for c in META_COLS_DEFAULT if c in df.columns]
    if "Symbol" in df.columns and "Symbol" not in meta:
        meta.insert(0, "Symbol")
    return meta


def get_qualifier_cols(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    ignore = set(meta_cols + ["__ticker_key__", "__symbol_key__"])
    return [c for c in df.columns if c not in ignore]


def build_symbol_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]

    out["__ticker_key__"] = out[symbol_col].astype(str).apply(normalize_ticker)
    out["__symbol_key__"] = out[symbol_col].astype(str).str.strip().str.upper()
    return out


def parse_manual_tickers(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\n,; \t]+", text)
    keys = [normalize_ticker(p) for p in parts if p]
    return list(dict.fromkeys(keys))


def clean_weight_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    s2 = pd.to_numeric(s2, errors="coerce")

    if not s2.dropna().empty:
        frac = (s2.dropna() <= 1).mean()
        if frac > 0.7:
            s2 = s2 * 100.0
    return s2


# ============================================================
# MATCHING ENGINE
# ============================================================

def find_memberships(
    df_indexed: pd.DataFrame,
    ticker_keys: List[str],
    qualifier_cols: List[str],
    match_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    long_rows = []
    matrix_rows: Dict[str, Dict[str, float]] = {}

    if match_mode == "Exact Symbol":
        key_col = "__symbol_key__"
        wanted = [k.upper() for k in ticker_keys]
    else:
        key_col = "__ticker_key__"
        wanted = ticker_keys

    for input_key in wanted:
        matches = df_indexed[df_indexed[key_col] == input_key]
        found = not matches.empty

        matrix_rows[input_key] = {}

        if found:
            for _, r in matches.iterrows():
                memberships = []
                for c in qualifier_cols:
                    v = r.get(c, None)
                    if pd.notna(v) and str(v).strip() != "":
                        memberships.append(c)
                        matrix_rows[input_key][c] = 1

                long_rows.append({
                    "Input": input_key,
                    "Matched Symbol": r.get("Symbol", ""),
                    "Found?": True,
                    "Membership Count": len(memberships),
                    "Membership Columns": ", ".join(memberships)
                })
        else:
            long_rows.append({
                "Input": input_key,
                "Matched Symbol": "",
                "Found?": False,
                "Membership Count": 0,
                "Membership Columns": ""
            })

    long_df = pd.DataFrame(long_rows)
    matrix = pd.DataFrame.from_dict(matrix_rows, orient="index").fillna(0)
    matrix.index.name = "Input"

    return long_df, matrix


# ============================================================
# EXPORT
# ============================================================

def to_excel_bytes(results_df, matrix_df, extra_sheets=None) -> bytes:
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Results")
        matrix_df.to_excel(writer, index=True, sheet_name="Matrix")

        if extra_sheets:
            for name, df in extra_sheets.items():
                df.to_excel(writer, index=False, sheet_name=name[:31])

    return bio.getvalue()


# ============================================================
# CHARTS
# ============================================================

def chart_pie(values, labels, title):
    fig, ax = plt.subplots(figsize=(5, 4))

    if sum(values) == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
    else:
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)

    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


# ============================================================
# UI
# ============================================================

st.title("Cross-Qualifying Ticker Finder")

with st.sidebar:
    st.header("Base File")
    base_upload = st.file_uploader("Upload base file", type=["csv", "xlsx", "xls"])
    base_path = st.text_input("Or use local path", value="Cross Qualifying Stocks.csv")

    st.divider()

    mode = st.radio("Mode", ["Ticker list", "Portfolio (with weights)"])

    st.divider()

    match_mode = st.radio("Match using", ["Ticker", "Exact Symbol"])
    show_only_found = st.toggle("Show only found tickers", value=False)

# Load base
base_df = load_base(base_path, base_upload)
base_df = build_symbol_index(base_df)

meta_cols = infer_meta_cols(base_df)
qualifier_cols = get_qualifier_cols(base_df, meta_cols)

ticker_keys = []
portfolio_df = None
weight_col = None

if mode == "Ticker list":
    manual = st.text_area("Paste tickers")
    ticker_keys = parse_manual_tickers(manual)

else:
    port_upload = st.file_uploader("Upload portfolio", type=["csv", "xlsx", "xls"])
    if port_upload:
        portfolio_df = load_table(port_upload)
        ticker_col = portfolio_df.columns[0]
        weight_col = portfolio_df.columns[1]

        portfolio_df["__ticker_key__"] = portfolio_df[ticker_col].apply(normalize_ticker)
        portfolio_df["__weight_pct__"] = clean_weight_series(portfolio_df[weight_col])

        ticker_keys = portfolio_df["__ticker_key__"].dropna().tolist()

if not ticker_keys:
    st.info("Add tickers to see results.")
    st.stop()

# ============================================================
# RESULTS
# ============================================================

long_results, matrix = find_memberships(
    base_df,
    ticker_keys,
    qualifier_cols,
    match_mode,
)

if show_only_found:
    long_results = long_results[long_results["Found?"]]

st.subheader("Results")
st.dataframe(long_results, use_container_width=True)

st.subheader("Summary by Column")
if not matrix.empty:
    summary = matrix.sum().sort_values(ascending=False).rename("Count")
    st.dataframe(summary.to_frame(), use_container_width=True)

st.subheader("Matrix")
st.dataframe(matrix, use_container_width=True)

# ============================================================
# DOWNLOAD
# ============================================================

st.divider()
csv_bytes = long_results.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, "results.csv")

xlsx_bytes = to_excel_bytes(long_results, matrix)
st.download_button(
    "Download Excel",
    xlsx_bytes,
    "results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
