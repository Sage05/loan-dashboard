import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from scipy import stats

st.set_page_config(layout="wide")

# =========================================================
# 🎨 PREMIUM UI
# =========================================================
st.markdown("""
<style>
body {background-color:#0b0f19; color:#e5e7eb;}
.metric-card {
    background: linear-gradient(145deg,#111827,#0f172a);
    padding:20px; border-radius:14px;
    box-shadow:0 0 25px rgba(0,255,255,0.08);
    text-align:center;
}
.metric-title {color:#9ca3af;}
.metric-value {font-size:28px; font-weight:bold; color:#22c55e;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA (SAFE + UNIQUE)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Ayuj Shingte\Desktop\DAL_LAB_CA\cleaned_loan_data.csv")

    cols = pd.Series(df.columns).astype(str).str.strip().str.lower().str.replace(" ", "_")

    seen = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)

    df.columns = new_cols
    return df

df = load_data()
filtered = df.copy()

st.title("💳 Loan Analytics Dashboard")

# =========================================================
# KPI
# =========================================================
num = filtered.select_dtypes(include=np.number)

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='metric-card'><div>Total Records</div><div class='metric-value'>{len(filtered)}</div></div>", unsafe_allow_html=True)

if len(num.columns) > 0:
    c2.markdown(f"<div class='metric-card'><div>Avg Value</div><div class='metric-value'>{round(num.mean().mean(),2)}</div></div>", unsafe_allow_html=True)

if len(num.columns) > 1:
    c3.markdown(f"<div class='metric-card'><div>Top Feature</div><div class='metric-value'>{num.mean().idxmax()}</div></div>", unsafe_allow_html=True)

if len(num.columns) > 0:
    c4.markdown(f"<div class='metric-card'><div>Max Value</div><div class='metric-value'>{round(num.max().max(),2)}</div></div>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "Overview",
    "Regression",
    "Correlation",
    "Clustering",
    "Boxplot",
    "Distribution",
    "Inference",
    "Insights",
    "Export"
])

# =========================================================
# OVERVIEW
# =========================================================
with tabs[0]:
    temp = filtered.copy()

    for col in temp.columns:
        try:
            temp[col] = pd.to_numeric(temp[col])
        except:
            pass

    num = temp.select_dtypes(include=np.number)

    if len(num.columns) >= 2:
        st.plotly_chart(px.scatter(temp, x=num.columns[0], y=num.columns[1], template="plotly_dark"), width="stretch", key="ov1")
        st.plotly_chart(px.histogram(temp, x=num.columns[0], template="plotly_dark"), width="stretch", key="ov2")
    else:
        st.dataframe(temp.head())

# =========================================================
# REGRESSION
# =========================================================
with tabs[1]:
    cols = filtered.select_dtypes(include=np.number).columns

    if len(cols) >= 2:
        x = st.selectbox("X Axis", cols, key="rx")
        y = st.selectbox("Y Axis", cols, key="ry")

        if x != y:
            temp = filtered[[x, y]].dropna()

            if len(temp) > 10:
                model = LinearRegression().fit(temp[[x]], temp[y])
                preds = model.predict(temp[[x]])

                fig = px.scatter(temp, x=x, y=y, template="plotly_dark")
                fig.add_scatter(x=temp[x], y=preds, mode="lines")

                st.plotly_chart(fig, width="stretch", key="reg")
                st.success(f"R² Score: {r2_score(temp[y], preds):.4f}")

# =========================================================
# CORRELATION
# =========================================================
with tabs[2]:
    num = filtered.select_dtypes(include=np.number)
    if len(num.columns) > 1:
        st.plotly_chart(px.imshow(num.corr(), text_auto=True, template="plotly_dark"), width="stretch", key="corr")

# =========================================================
# CLUSTERING
# =========================================================
with tabs[3]:
    cols = filtered.select_dtypes(include=np.number).columns
    selected = st.multiselect("Features", cols)

    if len(selected) >= 2:
        X = filtered[selected].dropna()
        if len(X) > 10:
            labels = KMeans(n_clusters=3).fit_predict(X)
            st.plotly_chart(px.scatter(x=X.iloc[:,0], y=X.iloc[:,1], color=labels, template="plotly_dark"), width="stretch", key="cluster")

# =========================================================
# BOXPLOT
# =========================================================
with tabs[4]:
    cols = filtered.select_dtypes(include=np.number).columns
    col = st.selectbox("Column", cols, key="box")
    st.plotly_chart(px.box(filtered, y=col, template="plotly_dark"), width="stretch", key="boxplot")

# =========================================================
# DISTRIBUTION
# =========================================================
with tabs[5]:
    cols = filtered.select_dtypes(include=np.number).columns
    col = st.selectbox("Column", cols, key="dist")
    st.plotly_chart(px.histogram(filtered, x=col, template="plotly_dark"), width="stretch", key="distplot")

# =========================================================
# INFERENCE
# =========================================================
with tabs[6]:
    cols = filtered.select_dtypes(include=np.number).columns
    col = st.selectbox("Column", cols, key="inf")

    d = filtered[col].dropna()
    if len(d) > 10:
        t, p = stats.ttest_ind(d.sample(len(d)//2), d.sample(len(d)//2))
        st.write("T-stat:", t)
        st.write("P-value:", p)

# =========================================================
# INSIGHTS
# =========================================================
with tabs[7]:
    num = filtered.select_dtypes(include=np.number)

    if len(num.columns) >= 2:
        corr = num.corr()
        pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False)

        top = pairs.index[0]
        st.markdown(f"### 🔗 Strongest Relationship\n{top[0]} ↔ {top[1]}")
        st.markdown(f"### 📉 Most Skewed\n{num.skew().abs().idxmax()}")
        st.markdown(f"### 📊 Highest Variance\n{num.var().idxmax()}")

# =========================================================
# EXPORT
# =========================================================
with tabs[8]:
    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Dataset",
        csv,
        "loan_data.csv",
        "text/csv"
    )