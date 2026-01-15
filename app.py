import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cross-Qualifying Ticker Finder", layout="wide")

# -----------------------------
# Constants / Matching
# -----------------------------
META_COLS_DEFAULT = ["Symbol", "Exchange", "Compagny", "Sector", "Rank"]
KNOWN_SUFFIXES = (".TO", ".V", ".CN", ".NE", ".TSX", ".TSXV")


def normalize_ticker(raw: str) -> str:
    """Normalize user-entered tickers/symbols to a comparable key."""
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    if not s:
        return ""

    # "XTSE:BTO" -> "BTO"
    if ":" in s:
        s = s.split(":")[-1].strip()

    # "BTO.TO" -> "BTO"
    for suf in KNOWN_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break

    # Remove whitespace/commas
    s = re.sub(r"[,\s]+", "", s)
    return s


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_table_from_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_base(base_path: str, uploaded_file=None) -> pd.DataFrame:
    """Load base universe from upload, or from repo file path."""
    if uploaded_file is not None:
        return load_table_from_upload(uploaded_file)

    p = Path(base_path)

    # Resolve relative paths from the app's directory (repo root)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p

    if not p.exists():
        raise FileNotFoundError(
            f"Base file not found: {p}. "
            f"Either add it to your repo (same folder as app.py) or upload it in the sidebar."
        )

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
# Parsing inputs
# -----------------------------
def parse_manual_tickers(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\n,; \t]+", text)
    keys = [normalize_ticker(p) for p in parts]
    keys = [k for k in keys if k]
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in df.columns:
        cl = str(c).strip().lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None


def parse_uploaded_ticker_file(uploaded) -> List[str]:
    if uploaded is None:
        return []
    tdf = load_table_from_upload(uploaded)
    if tdf.empty:
        return []

    col = infer_column(tdf, ["ticker", "tickers", "symbol", "symbols", "symbole", "symb"])
    if col is None:
        col = tdf.columns[0]

    keys = tdf[col].dropna().astype(str).tolist()
    keys = [normalize_ticker(x) for x in keys]
    keys = [k for k in keys if k]

    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# -----------------------------
# Core matching
# -----------------------------
def find_memberships(
    df_indexed: pd.DataFrame,
    ticker_keys: List[str],
    meta_cols: List[str],
    qualifier_cols: List[str],
    match_mode: str = "Ticker",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    if match_mode == "Exact Symbol":
        key_col = "__symbol_key__"
        wanted = [str(k).strip().upper() for k in ticker_keys]
    else:
        key_col = "__ticker_key__"
        wanted = ticker_keys

    matrix_rows: Dict[str, Dict[str, float]] = {}

    for input_key in wanted:
        if not input_key:
            continue

        matches = df_indexed[df_indexed[key_col] == input_key]
        if matches.empty:
            long_rows.append(
                {
                    "Input": input_key,
                    "Found?": False,
                    "Matched Symbol": "",
                    "Membership Columns": "",
                    "Membership Count": 0,
                }
            )
            continue

        matrix_rows.setdefault(input_key, {})
        for _, r in matches.iterrows():
            memberships = []
            for c in qualifier_cols:
                v = r.get(c, None)
                if pd.notna(v) and str(v).strip() != "":
                    memberships.append((c, v))
                    try:
                        matrix_rows[input_key][c] = max(matrix_rows[input_key].get(c, 0), float(v))
                    except Exception:
                        matrix_rows[input_key][c] = 1

            row = {
                "Input": input_key,
                "Found?": True,
                "Matched Symbol": r.get("Symbol", ""),
                "Membership Columns": ", ".join([m[0] for m in memberships]),
                "Membership Count": len(memberships),
            }
            for mc in meta_cols:
                if mc in matches.columns:
                    row[mc] = r.get(mc, "")
            long_rows.append(row)

    long_results = pd.DataFrame(long_rows)

    matrix = pd.DataFrame.from_dict(matrix_rows, orient="index")
    matrix.index.name = "Input"
    matrix = matrix.fillna(0)

    return long_results, matrix


# -----------------------------
# Portfolio helpers
# -----------------------------
def clean_weight_series(s: pd.Series) -> pd.Series:
    """Parse weights like '4.25' or '4.25%' or 0.0425 into percent."""
    s2 = s.copy()
    s2 = s2.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    s2 = pd.to_numeric(s2, errors="coerce")
    if s2.dropna().empty:
        return s2
    frac_share = (s2.dropna() <= 1).mean()
    if frac_share > 0.7:
        s2 = s2 * 100.0
    return s2


# -----------------------------
# Exports
# -----------------------------
def to_excel_bytes(
    results_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    extra_sheets: Optional[Dict[str, pd.DataFrame]] = None,
) -> bytes:
    import openpyxl
    from openpyxl.utils import get_column_letter

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Results")
        matrix_df.to_excel(writer, index=True, sheet_name="Matrix")

        if extra_sheets:
            for name, df in extra_sheets.items():
                df.to_excel(writer, index=False, sheet_name=name[:31])

        # Auto-fit widths
        for sheet_name in writer.book.sheetnames:
            ws = writer.book[sheet_name]
            for col_cells in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col_cells[0].column)
                for cell in col_cells:
                    val = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(val))
                ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    return bio.getvalue()


# -----------------------------
# Charts (matplotlib) - SAFE
# -----------------------------
def chart_pie(values: List[float], labels: List[str], title: str):
    # Matplotlib pie can crash when sum(values)==0 (produces NaNs)
    clean = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        if pd.isna(fv) or fv < 0:
            fv = 0.0
        clean.append(fv)

    total = sum(clean)
    if total <= 0:
        st.info("No usable weight values to plot the coverage pie (all weights are blank/0).")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.pie(
        clean,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=90,
    )
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


def chart_barh(df: pd.DataFrame, y_labels_col: str, x_values_col: str, title: str, top_n: int = 12):
    if df is None or df.empty:
        st.info("No data to chart.")
        return

    d = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.35 * len(d) + 1.2)))
    ax.barh(d[y_labels_col].astype(str), d[x_values_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x_values_col)
    st.pyplot(fig, clear_figure=True)


def chart_hist(values: pd.Series, title: str, bins: int = 10, xlabel: str = ""):
    if values is None or values.dropna().empty:
        st.info("No data to chart.")
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(values.dropna().astype(float), bins=bins)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# UI
# -----------------------------
st.title("Cross-Qualifying Ticker Finder")
st.caption(
    "Cross-reference a ticker list (or a weighted portfolio) to see which columns each ticker belongs to — based on your base file."
)

with st.sidebar:
    st.header("1) Base file")
    st.write(
        "Upload your base Excel/CSV (optional). If you don’t upload, the app uses the bundled file path below."
    )
    base_upload = st.file_uploader("Upload base file (.csv, .xlsx)", type=["csv", "xlsx", "xls"], key="base_upload")
    base_path = st.text_input("Bundled base path", value="Cross Qualifying Stocks.csv")

    st.divider()
    st.header("2) Mode")
    mode = st.radio("Choose input type", ["Ticker list", "Portfolio (with weights)"], index=0)

    st.divider()
    st.header("3) Options")
    match_mode = st.radio("Match using", ["Ticker", "Exact Symbol"], index=0)
    show_only_found = st.toggle("Show only found tickers", value=False)
    show_matrix_values = st.toggle("Matrix: show numeric values (when available)", value=True)

# Load base
try:
    base_df = load_base(base_path, uploaded_file=base_upload)
except Exception as e:
    st.error(str(e))
    st.stop()

base_df = build_symbol_index(base_df)
meta_cols = infer_meta_cols(base_df)
qualifier_cols = get_qualifier_cols(base_df, meta_cols)

ticker_keys: List[str] = []
portfolio_df = None

if mode == "Ticker list":
    with st.sidebar:
        st.header("4) Input tickers")
        manual = st.text_area("Paste tickers (comma/newline separated)", height=140)
        tickers_upload = st.file_uploader(
            "…or upload a ticker list (.csv, .xlsx)", type=["csv", "xlsx", "xls"], key="tickers_upload"
        )

    ticker_keys.extend(parse_manual_tickers(manual))
    ticker_keys.extend(parse_uploaded_ticker_file(tickers_upload))

else:
    with st.sidebar:
        st.header("4) Upload portfolio")
        port_upload = st.file_uploader("Upload portfolio (.csv, .xlsx)", type=["csv", "xlsx", "xls"], key="port_upload")
        if port_upload is not None:
            portfolio_df = load_table_from_upload(port_upload)

            ticker_guess = infer_column(portfolio_df, ["ticker", "symbol", "symbole", "ric"])
            weight_guess = infer_column(portfolio_df, ["weight", "cible", "allocation", "target", "%"])

            ticker_col = st.selectbox(
                "Ticker column",
                options=list(portfolio_df.columns),
                index=(list(portfolio_df.columns).index(ticker_guess) if ticker_guess in list(portfolio_df.columns) else 0),
            )
            weight_col = st.selectbox(
                "Weight column (%)",
                options=list(portfolio_df.columns),
                index=(list(portfolio_df.columns).index(weight_guess) if weight_guess in list(portfolio_df.columns) else min(1, len(portfolio_df.columns) - 1)),
                help="Accepts values like 4.25 or 4.25% or 0.0425 (auto-converts).",
            )

            portfolio_df = portfolio_df.copy()
            portfolio_df["__ticker_key__"] = portfolio_df[ticker_col].astype(str).apply(normalize_ticker)
            portfolio_df["__weight_pct__"] = clean_weight_series(portfolio_df[weight_col])

            ticker_keys = portfolio_df["__ticker_key__"].dropna().astype(str).tolist()
            ticker_keys = [t for t in ticker_keys if t]

# De-dupe tickers preserving order
seen = set()
ticker_keys = [t for t in ticker_keys if not (t in seen or seen.add(t))]

# Overview
col1, col2, col3 = st.columns([1.3, 1, 1])

with col1:
    st.subheader("Base dataset")
    st.write(f"Rows: **{len(base_df):,}** · Qualifier columns: **{len(qualifier_cols)}**")
    with st.expander("Preview base file"):
        preview_cols = [c for c in (meta_cols + qualifier_cols[:10]) if c in base_df.columns]
        st.dataframe(base_df[preview_cols].head(25), use_container_width=True)

with col2:
    st.subheader("Your input")
    st.write(f"Tickers provided: **{len(ticker_keys)}**")
    if ticker_keys:
        st.code("\n".join(ticker_keys[:30]) + ("\n…" if len(ticker_keys) > 30 else ""))

with col3:
    st.subheader("How membership works")
    st.write("Any non-empty value in a qualifier column counts as membership.")

st.divider()
st.subheader("Results")

if not ticker_keys:
    st.info("Add tickers (paste/upload) or upload a portfolio to see results.")
    st.stop()

long_results, matrix = find_memberships(
    df_indexed=base_df,
    ticker_keys=ticker_keys,
    meta_cols=meta_cols,
    qualifier_cols=qualifier_cols,
    match_mode=match_mode,
)

if show_only_found:
    long_results = long_results[long_results["Found?"] == True].copy()

# Attach weights in portfolio mode
extra_sheets: Dict[str, pd.DataFrame] = {}
if mode == "Portfolio (with weights)" and portfolio_df is not None:
    wmap = (
        portfolio_df.dropna(subset=["__ticker_key__"])
        .groupby("__ticker_key__", as_index=False)["__weight_pct__"]
        .sum()
        .rename(columns={"__ticker_key__": "Input", "__weight_pct__": "Weight (%)"})
    )
    long_results = long_results.merge(wmap, on="Input", how="left")

st.dataframe(long_results, use_container_width=True, hide_index=True)

# Summary
st.divider()
st.subheader("Summary (how many of your tickers appear in each column)")
presence = (matrix > 0).astype(int)
summary = presence.sum(axis=0).sort_values(ascending=False).rename("Count").to_frame()
st.dataframe(summary, use_container_width=True)

# Matrix view
st.divider()
st.subheader("Matrix view (tickers × columns)")
matrix_view = matrix.copy()
if not show_matrix_values:
    matrix_view = (matrix_view > 0).astype(int)
st.dataframe(matrix_view, use_container_width=True)

# Portfolio charts
if mode == "Portfolio (with weights)" and portfolio_df is not None:
    st.divider()
    st.header("Portfolio insights (weighted)")

    weights = (
        portfolio_df.dropna(subset=["__ticker_key__"])
        .groupby("__ticker_key__", as_index=False)["__weight_pct__"]
        .sum()
        .rename(columns={"__ticker_key__": "Input", "__weight_pct__": "Weight (%)"})
    )

    # If user picked the wrong weight column, everything can become NaN → stop charts cleanly
    if weights["Weight (%)"].dropna().empty or float(weights["Weight (%)"].fillna(0).sum()) <= 0:
        st.warning(
            "Your weights appear blank/0 after parsing. Double-check the selected Weight column in the sidebar "
            "(it should contain numbers or %)."
        )
        st.dataframe(weights, use_container_width=True, hide_index=True)
    else:
        mapped_inputs = set(matrix.index.tolist())
        weights["Mapped?"] = weights["Input"].apply(lambda x: x in mapped_inputs)

        mapped_w = float(weights.loc[weights["Mapped?"], "Weight (%)"].fillna(0).sum())
        unmapped_w = float(weights.loc[~weights["Mapped?"], "Weight (%)"].fillna(0).sum())
        total_w = float(weights["Weight (%)"].fillna(0).sum())

        a, b, c, d = st.columns(4)
        a.metric("Total weight (%)", f"{total_w:.2f}")
        b.metric("Mapped weight (%)", f"{mapped_w:.2f}")
        c.metric("Unmapped weight (%)", f"{unmapped_w:.2f}")
        d.metric("Coverage (%)", f"{(100 * mapped_w / total_w if total_w else 0):.1f}")

        left, right = st.columns([1, 1.2])
        with left:
            chart_pie([mapped_w, unmapped_w], ["Mapped", "Unmapped"], "Coverage by weight")
        with right:
            chart_hist(
                long_results.loc[long_results["Found?"] == True, "Membership Count"],
                "Membership Count distribution (mapped holdings)",
                bins=10,
                xlabel="Membership Count",
            )

        if not matrix.empty:
            mw = matrix.copy()
            mw = mw.merge(weights.set_index("Input")[["Weight (%)"]], left_index=True, right_index=True, how="left")
            mw["Weight (%)"] = mw["Weight (%)"].fillna(0)

            exp_rows = []
            for col in matrix.columns:
                m = mw[col] > 0
                exp_rows.append(
                    {"Column": col, "Weight (%)": float(mw.loc[m, "Weight (%)"].sum()), "Holdings (count)": int(m.sum())}
                )
            col_exp = pd.DataFrame(exp_rows).sort_values("Weight (%)", ascending=False)

            st.subheader("Top column exposures (by weight, mapped portion)")
            l1, l2 = st.columns([1.1, 1])
            with l1:
                st.dataframe(col_exp, use_container_width=True)
            with l2:
                chart_barh(col_exp, "Column", "Weight (%)", "Top columns by weight", top_n=12)

            extra_sheets["Column Exposure"] = col_exp

        unmapped = weights.loc[~weights["Mapped?"]].sort_values("Weight (%)", ascending=False)
        st.subheader("Unmapped tickers (not found in base file)")
        st.dataframe(unmapped, use_container_width=True, hide_index=True)
        extra_sheets["Unmapped (Portfolio)"] = unmapped

# Downloads
st.divider()
st.subheader("Download")

csv_bytes = long_results.to_csv(index=False).encode("utf-8")
st.download_button("Download results (CSV)", data=csv_bytes, file_name="cross_qualifying_results.csv", mime="text/csv")

xlsx_bytes = to_excel_bytes(long_results, matrix_view, extra_sheets=extra_sheets if extra_sheets else None)
st.download_button(
    "Download results (Excel)",
    data=xlsx_bytes,
    file_name="cross_qualifying_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
