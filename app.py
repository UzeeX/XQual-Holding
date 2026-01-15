import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cross-Qualifying Ticker Finder", layout="wide")

# -----------------------------
# Helpers
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

    # If they paste something like "XTSE:BTO" -> "BTO"
    if ":" in s:
        s = s.split(":")[-1].strip()

    # If they paste something like "BTO.TO" -> "BTO"
    for suf in KNOWN_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break

    # Remove obvious whitespace and stray commas
    s = re.sub(r"[,\s]+", "", s)

    return s

@st.cache_data(show_spinner=False)
def load_base(base_path: str, uploaded_file=None) -> pd.DataFrame:
    """Load base dataset from bundled file path or an uploaded file."""
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)

    p = Path(base_path)
    if not p.exists():
        raise FileNotFoundError(f"Base file not found: {base_path}")

    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)

def get_qualifier_cols(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in meta_cols]

def build_symbol_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns for matching."""
    out = df.copy()
    if "Symbol" in out.columns:
        out["__ticker_key__"] = out["Symbol"].astype(str).apply(normalize_ticker)
        out["__symbol_key__"] = out["Symbol"].astype(str).str.strip().str.upper()
    else:
        # Fallback if the base file has no 'Symbol' column
        first_col = out.columns[0]
        out["__ticker_key__"] = out[first_col].astype(str).apply(normalize_ticker)
        out["__symbol_key__"] = out[first_col].astype(str).str.strip().str.upper()
    return out

def parse_manual_tickers(text: str) -> List[str]:
    if not text:
        return []
    # Split by commas/newlines/semicolons/spaces
    parts = re.split(r"[\n,; \t]+", text)
    keys = [normalize_ticker(p) for p in parts]
    keys = [k for k in keys if k]
    # de-dupe preserving order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

def parse_uploaded_ticker_file(uploaded) -> List[str]:
    if uploaded is None:
        return []
    name = uploaded.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        tdf = pd.read_excel(uploaded)
    else:
        tdf = pd.read_csv(uploaded)

    if tdf.empty:
        return []

    # Prefer a column named like ticker/symbol
    cand_cols = [c for c in tdf.columns if str(c).strip().lower() in ("ticker", "tickers", "symbol", "symbols")]
    col = cand_cols[0] if cand_cols else tdf.columns[0]

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

def find_memberships(
    df_indexed: pd.DataFrame,
    ticker_keys: List[str],
    meta_cols: List[str],
    qualifier_cols: List[str],
    match_mode: str = "Ticker",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - long_results: one row per (input ticker, matched row) with membership list and count
      - matrix: boolean/value matrix (tickers x qualifiers) aggregated across matches
    """
    long_rows = []
    # Build quick lookup
    if match_mode == "Exact Symbol":
        key_col = "__symbol_key__"
        normalize = lambda x: str(x).strip().upper()
        wanted = [normalize(k) for k in ticker_keys]  # they likely entered full symbols
    else:
        key_col = "__ticker_key__"
        wanted = ticker_keys

    # For matrix, we aggregate presence across all matches for that ticker
    matrix_rows: Dict[str, Dict[str, float]] = {}

    for input_key in wanted:
        if not input_key:
            continue

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
                v = r.get(c, None)
                if pd.notna(v) and str(v).strip() != "":
                    memberships.append((c, v))
                    # Matrix: use 1 for presence, or numeric value if it parses
                    try:
                        matrix_rows[input_key][c] = max(matrix_rows[input_key].get(c, 0), float(v))
                    except Exception:
                        matrix_rows[input_key][c] = 1

            mem_cols = ", ".join([m[0] for m in memberships])
            row = {
                "Input": input_key,
                "Found?": True,
                "Matched Symbol": r.get("Symbol", ""),
                "Membership Columns": mem_cols,
                "Membership Count": len(memberships),
            }
            for mc in meta_cols:
                if mc in matches.columns:
                    row[mc] = r.get(mc, "")
            long_rows.append(row)

    long_results = pd.DataFrame(long_rows)

    # Build matrix dataframe
    matrix = pd.DataFrame.from_dict(matrix_rows, orient="index")
    matrix.index.name = "Input"
    matrix = matrix.fillna(0)

    return long_results, matrix

def to_excel_bytes(df1: pd.DataFrame, df2: pd.DataFrame) -> bytes:
    """Create an Excel with auto-fit columns for two sheets."""
    import openpyxl
    from openpyxl.utils import get_column_letter

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df1.to_excel(writer, index=False, sheet_name="Results")
        df2.to_excel(writer, index=True, sheet_name="Matrix")

        # Auto-fit widths
        for sheet_name in ["Results", "Matrix"]:
            ws = writer.book[sheet_name]
            for col_cells in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col_cells[0].column)
                for cell in col_cells:
                    val = "" if cell.value is None else str(cell.value)
                    if len(val) > max_len:
                        max_len = len(val)
                ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    return bio.getvalue()

# -----------------------------
# UI
# -----------------------------
st.title("Cross-Qualifying Ticker Finder")
st.caption("Paste or upload tickers → see which strategy/qualifier columns they appear in (based on your attached file).")

with st.sidebar:
    st.header("1) Base file")
    st.write("Upload your base Excel/CSV (optional). If you don’t upload, the app will use the bundled file path.")
    base_upload = st.file_uploader("Upload base file (.csv, .xlsx)", type=["csv", "xlsx", "xls"])
    base_path = st.text_input("Bundled base path", value="Cross Qualifying Stocks.csv")

    st.divider()

    st.header("2) Input tickers")
    manual = st.text_area(
        "Paste tickers (comma/newline separated)",
        placeholder="Example:\nBTO\nOTEX\nEFX\nor: XTSE:BTO, XTSE:OTEX",
        height=140,
    )
    tickers_upload = st.file_uploader("…or upload a ticker list (.csv, .xlsx)", type=["csv", "xlsx", "xls"], key="tickers_upload")

    st.divider()

    st.header("3) Options")
    match_mode = st.radio("Match using", ["Ticker", "Exact Symbol"], index=0, help="Ticker = matches 'BTO' to 'XTSE:BTO'. Exact Symbol expects full symbols like 'XTSE:BTO'.")
    show_matrix_values = st.toggle("Matrix: show numeric values (when available)", value=True)
    show_only_found = st.toggle("Show only found tickers", value=False)

# Load base
try:
    base_df = load_base(base_path, uploaded_file=base_upload)
except Exception as e:
    st.error(f"Could not load base file. Details: {e}")
    st.stop()

base_df = build_symbol_index(base_df)

# Infer meta cols present
meta_cols = [c for c in META_COLS_DEFAULT if c in base_df.columns]
if "Symbol" not in meta_cols and "Symbol" in base_df.columns:
    meta_cols = ["Symbol"] + meta_cols

qualifier_cols = get_qualifier_cols(base_df, meta_cols + ["__ticker_key__", "__symbol_key__"])

# Parse input tickers
ticker_keys = []
ticker_keys.extend(parse_manual_tickers(manual))
ticker_keys.extend(parse_uploaded_ticker_file(tickers_upload))

# De-dupe preserving order
seen = set()
ticker_keys = [t for t in ticker_keys if not (t in seen or seen.add(t))]

col1, col2, col3 = st.columns([1.3, 1, 1])

with col1:
    st.subheader("Base dataset")
    st.write(f"Rows: **{len(base_df):,}** · Qualifier columns: **{len(qualifier_cols)}**")
    with st.expander("Preview base file"):
        st.dataframe(base_df[[c for c in (meta_cols + qualifier_cols[:8]) if c in base_df.columns]].head(25), use_container_width=True)

with col2:
    st.subheader("Your input")
    st.write(f"Tickers provided: **{len(ticker_keys)}**")
    if ticker_keys:
        st.code("\n".join(ticker_keys[:30]) + ("\n…" if len(ticker_keys) > 30 else ""))

with col3:
    st.subheader("Column glossary")
    st.write("Any non-empty value in a qualifier column counts as membership.")
    st.write("Example: a '2' or '7' typically indicates ranking/position within that list.")

st.divider()
st.subheader("Results")

if not ticker_keys:
    st.info("Add tickers in the sidebar (paste or upload) to see results.")
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

# Show long results
st.dataframe(long_results, use_container_width=True, hide_index=True)

# Summary
st.divider()
st.subheader("Summary (how many of your tickers appear in each column)")

# For summary counts, treat presence as >0
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

# Downloads
st.divider()
st.subheader("Download")
csv_bytes = long_results.to_csv(index=False).encode("utf-8")
st.download_button("Download results (CSV)", data=csv_bytes, file_name="cross_qualifying_results.csv", mime="text/csv")

xlsx_bytes = to_excel_bytes(long_results, matrix_view)
st.download_button("Download results (Excel)", data=xlsx_bytes, file_name="cross_qualifying_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
