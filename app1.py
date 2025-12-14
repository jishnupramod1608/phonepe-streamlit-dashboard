# app.py
"""
Modern PhonePe / UPI Transaction Analysis Dashboard (Streamlit)
Drop this file into your project folder and run:
    streamlit run app.py
"""

import streamlit as st
# app.py
"""
Modern PhonePe / UPI dashboard with colourful charts & informative violin.
Run with:  streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px

# ------------------------------
# Streamlit basic config
# ------------------------------
st.set_page_config(
    page_title="PhonePe Transaction Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# GLOBAL COLOUR PALETTE & HELPERS
# ------------------------------
COLOR_PALETTE = [
    "#EF476F",  # red / pink
    "#FFD166",  # yellow
    "#06D6A0",  # green
    "#118AB2",  # blue
    "#073B4C",  # navy
    "#7C4DFF",  # purple
    "#FF7A00",  # orange
    "#00B4D8",  # cyan
    "#8AC926",  # lime
    "#FF006E",  # magenta
]

px.defaults.color_discrete_sequence = COLOR_PALETTE


def make_color_map(series: pd.Series) -> dict:
    """Map each unique value in a series to a colour from the palette."""
    unique_vals = series.dropna().unique().tolist()
    return {val: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, val in enumerate(unique_vals)}


# ------------------------------
# Global CSS (background & cards)
# ------------------------------
st.markdown(
    """
<style>
/* soft app background */
.stApp {
    background: linear-gradient(180deg, #f3f7fb 0%, #eef3f8 45%, #f9fbfc 100%) !important;
    font-family: "Segoe UI", Roboto, Arial, sans-serif;
}

/* main container */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f7f9fc) !important;
    border-right: 1px solid rgba(15,23,42,0.08);
}

/* header banner */
.header-banner {
    background: linear-gradient(90deg, #6c5ce7 0%, #00b4d8 100%);
    color: white !important;
    padding: 24px 26px;
    border-radius: 14px;
    margin-bottom: 18px;
    box-shadow: 0 10px 32px rgba(15,23,42,0.18);
}

/* cards & charts */
.stChart, .stDataFrame, .element-container div[data-testid="stMetricValue"] {
    border-radius: 10px;
}

/* metrics row spacing */
.css-12w0qpk, .css-1r6slb0 {
    gap: 0.75rem;
}

/* table row hover */
tbody tr:hover {
    background-color: rgba(15, 118, 110, 0.04) !important;
}

/* headings */
h1 {
    font-weight: 800;
    letter-spacing: -0.6px;
    color: #111827;
}
h2, h3 {
    color: #111827;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------
# Data loading helpers
# ------------------------------
@st.cache_data
def try_load_sample() -> pd.DataFrame | None:
    """Try to load a sample file from common paths (for notebook / demo)."""
    candidates = [
        "/mnt/data/PhonePe transactions.xlsx",
        "/mnt/data/PhonePe transactions (1).xlsx",
        "/mnt/data/PhonePe transactions.csv",
    ]
    for path in candidates:
        try:
            if path.lower().endswith(".xlsx"):
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)
        except Exception:
            continue
    return None


def load_uploaded(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return try_load_sample()
    if uploaded.name.lower().endswith(".xlsx"):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def extract_party(detail: str) -> str | None:
    if pd.isna(detail):
        return None
    s = str(detail).strip()
    prefixes = [
        "Paid to ",
        "Paid To ",
        "Paid ",
        "Received from ",
        "Received From ",
        "Refund from ",
        "Refund From ",
        "Refund ",
    ]
    for pre in prefixes:
        if s.startswith(pre):
            return s[len(pre) :].strip()
    return s


def simple_category(party: str) -> str:
    if pd.isna(party):
        return "Other"
    s = str(party).lower()
    if any(k in s for k in ["zomato", "swiggy", "restaurant", "cafe", "pizza", "food", "dine"]):
        return "Food"
    if any(k in s for k in ["mart", "grocery", "bigbasket", "supermarket", "store", "grocer", "greenmart"]):
        return "Groceries"
    if any(k in s for k in ["uber", "ola", "taxi", "cab", "auto"]):
        return "Transport"
    if any(k in s for k in ["recharge", "bill", "bills", "payment", "electricity", "mobile", "upi payment"]):
        return "Bills"
    if "refund" in s:
        return "Refund"
    # shortish names -> probably people
    if len(s.split()) <= 3 and any(c.isalpha() for c in s):
        return "Person"
    return "Other"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # amount column
    amt_col = None
    for name in ["amount", "Amount", "AMOUNT", "txn_amount", "Amount (INR)"]:
        if name in df.columns:
            amt_col = name
            break
    if amt_col is None:
        num_cols = df.select_dtypes(include=["number"]).columns
        amt_col = num_cols[0] if len(num_cols) else None
    df["amount"] = pd.to_numeric(df[amt_col], errors="coerce") if amt_col else np.nan

    # details
    details_col = None
    for name in ["Details", "details", "Transaction Details", "Description", "transaction details"]:
        if name in df.columns:
            details_col = name
            break
    if details_col:
        df["details"] = df[details_col].astype(str)
    else:
        df["details"] = df.iloc[:, 1].astype(str) if df.shape[1] > 1 else ""

    df["party"] = df["details"].apply(extract_party)
    df["category"] = df["party"].apply(simple_category)

    # type
    type_col = None
    for name in ["Type", "type", "Transaction Type", "Txn Type"]:
        if name in df.columns:
            type_col = name
            break
    df["type"] = df[type_col].astype(str).str.capitalize() if type_col else "Unknown"

    # time features
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date
    df["weekday"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.weekday >= 5
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return df


# ------------------------------
# SIDEBAR: upload + filters
# ------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your transactions (.xlsx or .csv)", type=["xlsx", "csv"]
)
raw_df = load_uploaded(uploaded_file)

if raw_df is None:
    st.sidebar.info("No file loaded. Upload a file to continue.")
    st.stop()

df = preprocess(raw_df)

with st.sidebar.expander("Filters", expanded=True):
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    start_date, end_date = st.date_input("Date range", value=(min_date, max_date))

    type_options = ["All"] + sorted(df["type"].dropna().unique().tolist())
    selected_type = st.selectbox("Transaction type", type_options)

    cat_options = sorted(df["category"].dropna().unique().tolist())
    selected_cats = st.multiselect("Categories", cat_options, default=cat_options)

    # top parties only, to keep dropdown usable
    top_parties = (
        df.groupby("party")["amount"].sum().sort_values(ascending=False).head(200).index.tolist()
    )
    party_options = ["All"] + top_parties
    selected_party = st.selectbox("Receiver (Top 200)", party_options)

# apply filters
mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
if selected_type != "All":
    mask &= df["type"] == selected_type
if selected_cats:
    mask &= df["category"].isin(selected_cats)
if selected_party != "All":
    mask &= df["party"] == selected_party

fdf = df[mask].copy()

if fdf.empty:
    st.warning("No data for selected filters. Try widening the date range or categories.")
    st.stop()

# ------------------------------
# HEADER BANNER
# ------------------------------
today_str = dt.date.today().strftime("%Y-%m-%d")
st.markdown(
    f"""
<div class="header-banner">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <h1 style="margin:0;">📊 PhonePe Transaction Insights</h1>
      <div style="margin-top:4px; font-size:15px; color:rgba(255,255,255,0.95);">
        Interactive dashboard – colourful charts, trends and key statistics.
      </div>
      <div style="margin-top:4px; font-size:13px; color:rgba(255,255,255,0.9);">
        Use the filters on the left; hover on charts for details.
      </div>
    </div>
    <div style="text-align:right; font-size:13px; color:rgba(255,255,255,0.95);">
      <div><b>Project:</b> Data Analysis</div>
      <div><b>Owner:</b> Jishnu</div>
      <div><b>Date:</b> {today_str}</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

# ------------------------------
# KPIs
# ------------------------------
def kpi_stats(df_: pd.DataFrame) -> dict:
    total_tx = len(df_)
    total_val = df_["amount"].sum()
    median_val = df_["amount"].median()
    mean_val = df_["amount"].mean()
    q99 = df_["amount"].quantile(0.99)
    top1_share = df_[df_["amount"] >= q99]["amount"].sum() / total_val if total_val else 0
    return {
        "total_tx": total_tx,
        "total_val": total_val,
        "median": median_val,
        "mean": mean_val,
        "q99": q99,
        "top1_share": top1_share,
    }


stats_dict = kpi_stats(fdf)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{stats_dict['total_tx']:,}")
c2.metric("Total value (₹)", f"{stats_dict['total_val']:.2f}")
c3.metric("Median amount (₹)", f"{stats_dict['median']:.2f}")
c4.metric("Mean amount (₹)", f"{stats_dict['mean']:.2f}")

# simple month delta
monthly = fdf.groupby("year_month")["amount"].sum().reset_index()
if not monthly.empty:
    last_val = monthly["amount"].iloc[-1]
    prev_val = monthly["amount"].iloc[-2] if len(monthly) > 1 else 0.0
    delta = last_val - prev_val
    st.write(
        f"**This month:** ₹{last_val:,.2f} — Change vs previous month: "
        f"{'+' if delta >= 0 else ''}{delta:,.2f}"
    )

st.markdown("---")

# ------------------------------
# Monthly stacked trend & Top receivers
# ------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Monthly trend (stacked by category)")
    monthly_cat = (
        fdf.groupby(["year_month", "category"])["amount"]
        .sum()
        .reset_index()
        .sort_values("year_month")
    )
    if not monthly_cat.empty:
        fig_month = px.area(
            monthly_cat,
            x="year_month",
            y="amount",
            color="category",
            color_discrete_map=make_color_map(monthly_cat["category"]),
            labels={"year_month": "Month", "amount": "Amount (₹)"},
        )
        fig_month.update_layout(hovermode="x unified", legend_title_text="Category")
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.info("Not enough data to show monthly trend.")

with right:
    st.subheader("Top receivers (by amount)")
    top_recv = (
        fdf.groupby("party")["amount"].sum().sort_values(ascending=False).head(15).reset_index()
    )
    st.dataframe(top_recv, use_container_width=True, height=360)
    fig_top = px.bar(
        top_recv.sort_values("amount"),
        x="amount",
        y="party",
        orientation="h",
        color="party",
        color_discrete_map=make_color_map(top_recv["party"]),
        labels={"amount": "Amount (₹)", "party": "Receiver"},
    )
    fig_top.update_layout(showlegend=False)
    st.plotly_chart(fig_top, use_container_width=True)

st.markdown("---")

# ------------------------------
# Category treemap & weekday/month heatmap
# ------------------------------
ct1, ct2 = st.columns(2)

with ct1:
    st.subheader("Category share (treemap)")
    cat_sum = (
        fdf.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
    )
    if not cat_sum.empty:
        fig_tree = px.treemap(
            cat_sum,
            path=["category"],
            values="amount",
            color="category",
            color_discrete_map=make_color_map(cat_sum["category"]),
        )
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No category data.")

with ct2:
    st.subheader("Weekday × Month heatmap (amount)")
    pivot = (
        fdf.groupby([fdf["date"].dt.to_period("M").astype(str), "weekday"])["amount"]
        .sum()
        .unstack(fill_value=0)
    )
    # order weekdays
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot[order] if set(order).issubset(pivot.columns) else pivot
    if not pivot.empty:
        fig_heat = px.imshow(
            pivot.T,
            x=pivot.index,
            y=pivot.columns,
            labels={"x": "Month", "y": "Weekday", "color": "Amount (₹)"},
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")

st.markdown("---")

# ------------------------------
# Distribution + INFORMATIVE VIOLIN
# ------------------------------
st.subheader("Amount distribution")

hcol, vcol = st.columns([2, 2])

with hcol:
    log_opt = st.checkbox("Use log scale for histogram", value=False)
    if log_opt:
        fdf["log_amount"] = np.log1p(fdf["amount"].clip(lower=0))
        fig_hist = px.histogram(
            fdf,
            x="log_amount",
            nbins=50,
            color="category",
            color_discrete_map=make_color_map(fdf["category"]),
            labels={"log_amount": "log(Amount + 1)", "count": "Count"},
        )
    else:
        fig_hist = px.histogram(
            fdf,
            x="amount",
            nbins=50,
            color="category",
            color_discrete_map=make_color_map(fdf["category"]),
            labels={"amount": "Amount (₹)", "count": "Count"},
        )
    fig_hist.update_layout(bargap=0.02)
    st.plotly_chart(fig_hist, use_container_width=True)

with vcol:
    st.markdown("**Amount by category — informative violin**")

    top_cats = fdf["category"].value_counts().index[:6].tolist()
    vdf = fdf[fdf["category"].isin(top_cats)].copy()

    if vdf.empty:
        st.write("Not enough data to draw violin plot.")
    else:
        if len(vdf) > 4000:
            vdf = vdf.sample(4000, random_state=42)

        cmap = make_color_map(vdf["category"])

        fig_v = px.violin(
            vdf,
            x="category",
            y="amount",
            color="category",
            box=True,
            points="outliers",
            color_discrete_map=cmap,
            hover_data=["party", "date"],
            labels={"amount": "Amount (₹)", "category": "Category"},
        )
        fig_v.update_traces(meanline_visible=True, spanmode="hard")
        for tr in fig_v.data:
            tr.update(
                fillcolor=tr.marker.color,
                opacity=0.35,
                line=dict(color=tr.marker.color, width=2),
            )
        fig_v.update_layout(showlegend=False, margin=dict(t=30, b=0))
        fig_v.update_yaxes(tickformat=",", title_text="Amount (₹)")

        # stats table for non-technical users
        stats_table = (
            vdf.groupby("category")["amount"]
            .agg(
                Txn_count="count",
                Median=lambda x: round(x.median(), 2),
                Mean=lambda x: round(x.mean(), 2),
                P90=lambda x: round(np.percentile(x, 90), 2),
            )
            .reset_index()
            .sort_values("Median", ascending=False)
            .set_index("category")
        )

        v_left, v_right = st.columns([3, 2])
        with v_left:
            st.plotly_chart(fig_v, use_container_width=True)
        with v_right:
            st.markdown("**Category statistics**")
            st.dataframe(stats_table, height=260)
            st.markdown(
                """
**How to read this (simple):**
- Wider parts of the shape = more transactions around those amounts.  
- Box inside shows the middle 50% of transactions (typical range).  
- Mean & median in the table help compare typical spend across categories.  
- 90th percentile (P90) shows what counts as a *large* transaction.
"""
            )

st.markdown("---")

# ------------------------------
# Outliers & top 1%
# ------------------------------
st.subheader("Outliers & top 1% transactions")

if len(fdf) > 0:
    q99_val = fdf["amount"].quantile(0.99)
    top1 = fdf[fdf["amount"] >= q99_val].sort_values("amount", ascending=False)
    o1, o2 = st.columns([1, 2])
    with o1:
        st.write(f"99th percentile: **₹{q99_val:.2f}**")
        st.write(f"Top 1% count: **{len(top1):,}**")
        share = (
            top1["amount"].sum() / fdf["amount"].sum()
            if fdf["amount"].sum() > 0
            else 0
        )
        st.write(f"Top 1% share of total value: **{share:.2%}**")
    with o2:
        st.dataframe(
            top1[["date", "party", "category", "type", "amount"]].head(15),
            use_container_width=True,
        )
else:
    st.write("No data to analyse outliers.")

st.markdown("---")

# ------------------------------
# Hypothesis tests
# ------------------------------
st.subheader("Quick hypothesis checks")

if len(fdf) >= 30:
    # weekend vs weekday
    week_vals = fdf[~fdf["is_weekend"]]["amount"].dropna()
    weekend_vals = fdf[fdf["is_weekend"]]["amount"].dropna()
    if len(week_vals) >= 10 and len(weekend_vals) >= 10:
        u_stat, p_val = stats.mannwhitneyu(
            weekend_vals, week_vals, alternative="two-sided"
        )
        st.write(
            f"• Weekend vs weekday transaction amounts (Mann–Whitney): "
            f"p = **{p_val:.4f}**, median weekend = ₹{weekend_vals.median():.2f}, "
            f"weekday = ₹{week_vals.median():.2f}"
        )

    # debit vs credit
    has_dc = fdf["type"].str.lower().isin(["debit", "credit"]).any()
    if has_dc:
        dvals = fdf[fdf["type"].str.lower() == "debit"]["amount"].dropna()
        cvals = fdf[fdf["type"].str.lower() == "credit"]["amount"].dropna()
        if len(dvals) >= 10 and len(cvals) >= 10:
            t_stat, p_t = stats.ttest_ind(
                dvals, cvals, equal_var=False, nan_policy="omit"
            )
            st.write(
                f"• Debit vs credit amounts (t-test): "
                f"p = **{p_t:.4f}**, mean debit = ₹{dvals.mean():.2f}, "
                f"mean credit = ₹{cvals.mean():.2f}"
            )
else:
    st.write("Not enough rows for reliable statistical tests (need ~30+).")

st.markdown("---")

# ------------------------------
# Suggested key insights for report
# ------------------------------
st.subheader("Key insights")

insights = [
    f"Total transactions analysed: **{stats_dict['total_tx']:,}**, "
    f"total value **₹{stats_dict['total_val']:.2f}**.",
    f"Typical transaction size: median **₹{stats_dict['median']:.2f}**, "
    f"mean **₹{stats_dict['mean']:.2f}**.",
    f"Top 1% of transactions account for roughly **{stats_dict['top1_share']:.2%}** "
    "of total value, showing a heavy-tailed spending pattern.",
    "Treemap & pie charts indicate which categories dominate overall spending "
    "(e.g., Groceries / Bills / Person transfers).",
    "Monthly stacked chart shows how category-wise spending evolves over time, "
    "with visible peaks and dips.",
    "Weekday–month heatmap reveals which days of week have higher spending intensity.",
    "Violin + stats table compare distributions across categories, highlighting which "
    "categories have higher typical and high-end (P90) transaction values.",
]

for line in insights:
    st.markdown(f"- {line}")
# ---------------------------
# End
# ---------------------------