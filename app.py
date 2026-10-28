# ==========================================================
# Government Fund Utilization Prediction Dashboard
# Developed by Nirali Banubakode
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Government Fund Utilization Prediction",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------
# Load CSS
# ----------------------------------------------------------

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("fund_allocation.csv")

df = load_data()

# ----------------------------------------------------------
# Load ML Model
# ----------------------------------------------------------

model = joblib.load("fund_utilization_model.pkl")
scheme_encoder = joblib.load("scheme_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")

# ----------------------------------------------------------
# Dashboard Header
# ----------------------------------------------------------

st.markdown("""
<div style="text-align:center;padding:20px;">
<h1>🏛️ Government Fund Utilization Prediction Dashboard</h1>

<h4>
Predict Government Fund Utilization using Machine Learning
</h4>

</div>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

rows = df.shape[0]
columns = df.shape[1]

st.write(f"### Dataset Shape : {rows:,} Rows × {columns} Columns")
# ==========================================================
# KPI SECTION
# ==========================================================

st.markdown("## 📊 Dashboard Overview")

total_records = len(df)
total_schemes = df["Scheme_ID"].nunique()
total_states = df["State"].nunique()
avg_utilization = df["Fund_Utilized"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📄 Total Records",
        value=f"{total_records:,}"
    )

with col2:
    st.metric(
        label="🏛️ Total Schemes",
        value=total_schemes
    )

with col3:
    st.metric(
        label="🌍 States",
        value=total_states
    )

with col4:
    st.metric(
        label="💰 Avg Fund Utilized",
        value=f"{avg_utilization:.2f}"
    )

st.divider()

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.title("⚙️ Dashboard Filters")

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(df["State"].unique().tolist())
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All"] + sorted(df["Year"].unique().tolist())
)

filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["State"] == selected_state
    ]

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["Year"] == selected_year
    ]

# ==========================================================
# DATASET PREVIEW
# ==========================================================

st.markdown("## 📑 Dataset Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True,
    hide_index=True
)

st.success(
    f"Showing {len(filtered_df):,} records."
)

st.divider()
# ==========================================================
# INTERACTIVE VISUALIZATIONS
# ==========================================================

st.markdown("## 📈 Interactive Dashboard")

# ---------- Row 1 ----------

col1, col2 = st.columns(2)

with col1:

    state_data = (
        filtered_df.groupby("State")["Fund_Utilized"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        state_data,
        x="State",
        y="Fund_Utilized",
        color="Fund_Utilized",
        title="Fund Utilized by State"
    )

    fig.update_layout(
        xaxis_title="State",
        yaxis_title="Fund Utilized",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    year_data = (
        filtered_df.groupby("Year")["Fund_Utilized"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        year_data,
        x="Year",
        y="Fund_Utilized",
        markers=True,
        title="Fund Utilization Trend"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Row 2 ----------

col3, col4 = st.columns(2)

with col3:

    pie = (
        filtered_df.groupby("State")["Fund_Allocated"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        pie,
        names="State",
        values="Fund_Allocated",
        title="Fund Allocation Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col4:

    fig = px.scatter(
        filtered_df,
        x="Fund_Allocated",
        y="Fund_Utilized",
        color="State",
        hover_data=["Year"],
        title="Allocated vs Utilized Fund"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

st.markdown("## 🔥 Correlation Heatmap")

corr = filtered_df.select_dtypes(include=np.number).corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Viridis",
    title="Correlation Matrix"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# SUMMARY TABLE
# ==========================================================

st.markdown("## 📋 Summary Statistics")

st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)
# ==========================================================
# FUND UTILIZATION PREDICTION
# ==========================================================

st.divider()

st.markdown("## 🤖 Fund Utilization Prediction")

col1, col2 = st.columns(2)

with col1:
    input_year = st.selectbox(
        "Select Year",
        sorted(df["Year"].unique())
    )

    input_scheme = st.selectbox(
        "Select Scheme",
        sorted(df["Scheme_ID"].unique())
    )

    input_state = st.selectbox(
        "Select State",
        sorted(df["State"].unique())
    )

with col2:
    input_allocated = st.number_input(
        "Fund Allocated",
        min_value=0,
        value=1000,
        step=100
    )

    input_released = st.number_input(
        "Fund Released",
        min_value=0,
        value=500,
        step=100
    )

st.write("")

if st.button("🚀 Predict Fund Utilization", use_container_width=True):

    # Create Input DataFrame
    input_df = pd.DataFrame({
        "Year": [input_year],
        "Scheme_ID": [input_scheme],
        "State": [input_state],
        "Fund_Allocated": [input_allocated],
        "Fund_Released": [input_released]
    })

    # Encode Categorical Columns
    input_df["Scheme_ID"] = scheme_encoder.transform(input_df["Scheme_ID"])
    input_df["State"] = state_encoder.transform(input_df["State"])

    # Prediction
    prediction = model.predict(input_df)[0]

    # Negative prediction avoid
    prediction = max(0, prediction)

    st.success("Prediction Completed Successfully ✅")

    st.metric(
        label="Predicted Fund Utilized",
        value=f"{prediction:.2f}"
    )

    st.progress(min(prediction / 5000, 1.0))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Predicted Fund Utilized"},
            gauge={
                "axis": {"range": [0, 5000]},
                "bar": {"color": "green"},
                "steps": [
                    {"range": [0, 1500], "color": "lightgray"},
                    {"range": [1500, 3000], "color": "gold"},
                    {"range": [3000, 5000], "color": "limegreen"},
                ],
            },
        )
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)
# ==========================================================
# DATA EXPLORER
# ==========================================================

st.divider()

st.markdown("## 📂 Data Explorer")

show_data = st.checkbox("Show Complete Dataset")

if show_data:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="filtered_fund_allocation.csv",
    mime="text/csv",
    use_container_width=True
)

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()

st.markdown("## 🧠 Model Information")

info1, info2 = st.columns(2)

with info1:

    st.info("""
**Best Machine Learning Model**

• Lasso Regression

**R² Score**

• 0.549

**Prediction Type**

• Regression

**Target**

• Fund Utilized
""")

with info2:

    st.success("""
**Input Features**

✔ Year

✔ Scheme ID

✔ State

✔ Fund Allocated

✔ Fund Released
""")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.divider()

st.markdown("## 📖 About Project")

st.write("""
This dashboard predicts Government Fund Utilization using Machine Learning.

The model was trained using historical government funding data.

Users can:

- Predict Fund Utilization
- Explore Dataset
- View Interactive Charts
- Filter Records
- Download Data
- Analyze State-wise Utilization

The prediction is generated using the trained Lasso Regression model.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
"""
<div style="text-align:center; padding:20px;">

<h4>🏛 Government Fund Utilization Prediction Dashboard</h4>

<p>
Developed using
<b>Python • Streamlit • Plotly • Machine Learning</b>
</p>

<p>
© 2026 Nirali Banubakode
</p>

</div>
""",
unsafe_allow_html=True
)
