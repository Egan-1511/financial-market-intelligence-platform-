import streamlit as st
import pandas as pd

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Financial AI",
    page_icon="📈",
    layout="wide"
)

# ==================================
# LOAD DATA
# ==================================

ranking = pd.read_csv(
    "predictions/stock_rankings.csv"
)

# ==================================
# SIDEBAR
# ==================================

with st.sidebar:

    st.title("📈 Financial AI")

    st.success("Model: XGBoost")

    st.metric(
        "ROC-AUC",
        "69.66%"
    )

    st.metric(
        "Stocks",
        "49"
    )

    st.metric(
        "Records",
        "121K+"
    )

    st.markdown("---")

    stock_search = st.selectbox(
        "🔍 Search Stock",
        ranking["Ticker"]
    )

# ==================================
# TITLE
# ==================================

st.title("🚀 AI-Powered Stock Ranking Platform")

st.markdown(
    """
    Advanced Financial Intelligence System powered by:

    - XGBoost
    - NIFTY Market Indicators
    - India VIX Features
    - 87 Engineered Features
    - 49 NSE Stocks
    """
)

# ==================================
# MARKET OVERVIEW
# ==================================

st.subheader("📊 Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "NIFTY RSI",
        "58"
    )

with col2:
    st.metric(
        "VIX RSI",
        "42"
    )

with col3:
    st.metric(
        "Market Trend",
        "Bullish"
    )

with col4:
    st.metric(
        "Top Pick",
        ranking.iloc[0]["Ticker"]
    )

# ==================================
# SIGNAL FUNCTION
# ==================================

def signal(prob):

    if prob >= 80:
        return "🟢 Strong Buy"

    elif prob >= 65:
        return "🟡 Buy"

    else:
        return "🔴 Watch"

# ==================================
# TOP STOCK CARDS
# ==================================

st.subheader("🏆 Top 10 Opportunities")

top10 = ranking.head(10)

for _, row in top10.iterrows():

    st.markdown(
        f"""
### {row['Ticker']}

💰 Price: ₹{row['Close']:.2f}

🎯 Probability: {row['Buy_Probability']:.2f}%

{signal(row['Buy_Probability'])}

---
"""
    )

# ==================================
# BUY PROBABILITY CHART
# ==================================

st.subheader("📈 Buy Probability Ranking")

chart_df = ranking.head(10)

st.bar_chart(
    chart_df.set_index("Ticker")[
        "Buy_Probability"
    ]
)

# ==================================
# STOCK SEARCH
# ==================================

st.subheader("🔍 Stock Analysis")

selected_stock = ranking[
    ranking["Ticker"] == stock_search
]

if not selected_stock.empty:

    row = selected_stock.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            f"₹{row['Close']:.2f}"
        )

    with col2:
        st.metric(
            "Buy Probability",
            f"{row['Buy_Probability']:.2f}%"
        )

    with col3:
        st.metric(
            "Signal",
            signal(row['Buy_Probability'])
        )

# ==================================
# SIGNAL DISTRIBUTION
# ==================================

st.subheader("📊 Recommendation Distribution")

strong_buy = len(
    ranking[
        ranking["Buy_Probability"] >= 80
    ]
)

buy = len(
    ranking[
        (ranking["Buy_Probability"] >= 65)
        &
        (ranking["Buy_Probability"] < 80)
    ]
)

watch = len(
    ranking[
        ranking["Buy_Probability"] < 65
    ]
)

distribution = pd.DataFrame(
    {
        "Category": [
            "Strong Buy",
            "Buy",
            "Watch"
        ],
        "Count": [
            strong_buy,
            buy,
            watch
        ]
    }
)

st.bar_chart(
    distribution.set_index(
        "Category"
    )
)

# ==================================
# FULL TABLE
# ==================================

st.subheader("📋 Complete Ranking")

st.dataframe(
    ranking,
    use_container_width=True
)

# ==================================
# MODEL INFO
# ==================================

st.subheader("🤖 Model Information")

st.info(
    """
Best Model: XGBoost

ROC-AUC: 69.66%

Dataset Size: 121,478 Records

Features: 87

Stocks Covered: 49

Market Features:
- NIFTY Indicators
- India VIX Indicators
- Technical Indicators
"""
)