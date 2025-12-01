import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go

# I am configuring the Streamlit page to use a wide layout and a custom title/icon
st.set_page_config(
    page_title="Bitcoin Price Explorer (2013–2017)",
    page_icon="🪙",
    layout="wide",
)

# I am injecting a bit of custom CSS to make the app look cleaner and more modern
st.markdown(
    """
    <style>
    /* I am tightening the main content width and centering it slightly */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: auto;
    }

    /* I am giving metric blocks a softer rounded look */
    .stMetric {
        background: #111827;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 0 10px rgba(15,23,42,0.5);
    }

    /* I am slightly rounding plots and elements */
    img, .element-container {
        border-radius: 0.75rem;
    }

    /* I am styling horizontal rules used in the footer */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, #4b5563, #9ca3af, #4b5563);
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# I am setting the default visual style for Matplotlib and Seaborn
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_theme(style="darkgrid")

# I am building the data path from the location of this file so it works locally and on Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent          # .../notebooks
DATA_DIR = BASE_DIR.parent / "data"                 # .../data
CSV_PATH = DATA_DIR / "bitcoin_price_training.csv"



@st.cache_data
def load_and_prepare_data():
    # I am loading the dataset from a relative path so that it works locally and on Streamlit Cloud
    df = pd.read_csv(CSV_PATH)

    # I am converting the Date column to datetime and dropping invalid rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # I am sorting the data in chronological order
    df = df.sort_values("Date").reset_index(drop=True)

    # I am removing duplicate rows, if any exist
    df = df.drop_duplicates().reset_index(drop=True)

    # I am setting Date as the index for resampling and time-series operations
    data = df.set_index("Date").copy()

    # I am calculating the daily percentage change in the closing price
    data["Close_pct_change"] = data["Close"].pct_change() * 100

    return df, data


def compute_summary_stats(df_filtered: pd.DataFrame, data_filtered: pd.DataFrame):
    # I am creating a helper function to calculate easy-to-read summary numbers
    stats = {
        "start_price": None,
        "end_price": None,
        "total_return_pct": None,
        "avg_daily_move_pct": None,
        "max_gain_pct": None,
        "max_gain_date": None,
        "max_drop_pct": None,
        "max_drop_date": None,
    }

    if len(df_filtered) < 2:
        return stats

    stats["start_price"] = df_filtered["Close"].iloc[0]
    stats["end_price"] = df_filtered["Close"].iloc[-1]
    stats["total_return_pct"] = (stats["end_price"] / stats["start_price"] - 1) * 100

    daily_change = data_filtered["Close_pct_change"].dropna()
    if not daily_change.empty:
        stats["avg_daily_move_pct"] = daily_change.abs().mean()
        stats["max_gain_pct"] = daily_change.max()
        stats["max_gain_date"] = daily_change.idxmax().date()
        stats["max_drop_pct"] = daily_change.min()
        stats["max_drop_date"] = daily_change.idxmin().date()

    return stats


def main():
    # ========= Sidebar =========
    st.sidebar.title("Bitcoin Price Analysis (2013–2017)")
    st.sidebar.markdown(
        """
I am exploring historical Bitcoin price data from **2013–2017**, 
focusing on trends, volatility and time-based patterns.

You can:
- Change the candlestick window  
- Choose how to group prices (year/quarter/month)  
- Adjust the date range in the main view  
        """
    )

    # I am letting the user control the candlestick sample size
    sample_days = st.sidebar.slider(
        "Number of recent days for candlestick sample",
        min_value=30,
        max_value=120,
        value=60,
        step=10,
    )

    # I am using YE/QE/ME because Y/Q/M are deprecated in pandas
    resample_freq = st.sidebar.selectbox(
    "Group prices by:",
    options=[("YE", "Yearly"), ("QE", "Quarterly"), ("ME", "Monthly")],
    index=0,
    format_func=lambda x: x[1],
)


    # ========= Load data =========
    df, data = load_and_prepare_data()

    # ========= Title & Hero / Overview =========
    st.title("Bitcoin Price Explorer (2013–2017)")

    # I am creating a header section that explains the app in simple language
    with st.container():
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.markdown(
                """
### 📌 Project Overview

I built this application to explore and explain how the price of **Bitcoin (BTC)** moved between **2013 and 2017** using clear visuals and easy-to-understand explanations.  
All prices shown in this app are in **US Dollars (USD)** for **one Bitcoin (1 BTC)**.

Bitcoin is known for being highly volatile, meaning its price can rise or fall very quickly.  
This app helps the user understand **how big these movements were**, **when they happened**, and **how the market behaved over time**.
                """
            )

        with right_col:
            st.markdown(
                """
**What value does this app give you?**

- See long-term price trends  
- Analyse daily price movements (Open, High, Low, Close)  
- Explore volatility in plain language  
- Compare **Yearly / Quarterly / Monthly** averages  
- Inspect candlestick charts for recent periods  
- Interact with the data by changing the time range

This is useful for beginners, students and analysts who want to understand Bitcoin’s past behaviour without heavy financial jargon.
                """
            )

    # ========= Date Range Filter =========
    st.subheader("1. Choose a time period")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    # I am using a slider to let the user focus on a specific period
    start_date, end_date = st.slider(
        "I am selecting the analysis window:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()

    # I am creating a filtered time-series dataframe for the selected period
    data_filtered = df_filtered.set_index("Date").copy()

    # I am recalculating the daily percentage change for this filtered period
    data_filtered["Close_pct_change"] = data_filtered["Close"].pct_change() * 100

    st.info(
        f"I am currently analysing data from **{start_date}** to **{end_date}** "
        f"covering **{len(df_filtered)} days**."
    )

    # ========= Summary metrics =========
    st.subheader("2. Quick summary for the selected period")

    stats = compute_summary_stats(df_filtered, data_filtered)

    if len(df_filtered) < 2:
        st.warning("I need at least two days of data to calculate summary stats. Please widen the date range.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Start vs End Price (USD)",
            f"${stats['start_price']:.2f} → ${stats['end_price']:.2f}",
        )

        col2.metric(
            "Total Price Change over Period",
            f"{stats['total_return_pct']:.1f}%",
            help="This shows how much the closing price changed between the first and last day in the selected range.",
        )

        if stats["avg_daily_move_pct"] is not None:
            col3.metric(
                "Average Daily Move",
                f"{stats['avg_daily_move_pct']:.2f}%",
                help="On an average day, the price moved up or down by about this percentage.",
            )

        st.markdown(
            """
I am using these numbers to give a plain-English view:

- **Total Price Change** tells me how much Bitcoin grew or fell over the selected period.  
- **Average Daily Move** shows how “shaky” the price was on a typical day.  
If this number is high, it means the price jumped a lot from one day to the next.
            """
        )

        if stats["max_gain_pct"] is not None:
            st.markdown(
                f"""
- The **biggest one-day jump** was about **{stats['max_gain_pct']:.2f}%** on **{stats['max_gain_date']}**.  
- The **biggest one-day drop** was about **{stats['max_drop_pct']:.2f}%** on **{stats['max_drop_date']}**.  

These extreme days are useful for understanding risk: they show how quickly the price could move in a single day.
                """
            )

    # ========= Tabs for detailed analysis =========
    tab_overview, tab_charts, tab_volatility, tab_data = st.tabs(
        ["Overview charts", "Detailed price charts", "Volatility view", "About the data"]
    )

    # ======== TAB 1: Overview charts ========
    with tab_overview:
        st.markdown("### Overall closing price trend")

        fig_trend, ax_trend = plt.subplots(figsize=(12, 4))
        ax_trend.plot(df_filtered["Date"], df_filtered["Close"])
        ax_trend.set_title("Bitcoin Closing Price over Time")
        ax_trend.set_xlabel("Date")
        ax_trend.set_ylabel("Price (USD)")
        plt.tight_layout()
        st.pyplot(fig_trend)

        st.caption(
            "Here I am showing the closing price each day. "
            "This helps the user see the overall direction: long flat periods, sharp rises, or deep drops."
        )

        st.markdown("### Linear vs Logarithmic scale")

        fig_scale, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # I am plotting the raw closing price on a linear scale
        ax1.plot(data_filtered.index, data_filtered["Close"])
        ax1.set_title("Closing Price (Linear Scale)")
        ax1.set_ylabel("Price (USD)")

        # I am applying a log transform to better visualise large percentage changes
        ax2.plot(data_filtered.index, np.log1p(data_filtered["Close"]))
        ax2.set_title("Closing Price (Log Scale)")
        ax2.set_ylabel("log(1 + Price)")

        plt.tight_layout()
        st.pyplot(fig_scale)

        st.caption(
            "On the left I am using the normal price scale. On the right I am using a log scale. "
            "The log scale makes early years and later years easier to compare when prices grow a lot."
        )

    # ======== TAB 2: Detailed price charts ========
    with tab_charts:
        st.markdown("### OHLC price lines")

        price_cols = ["Open", "High", "Low", "Close"]
        fig_ohlc, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

        # I am plotting Open, High, Low and Close in a 2x2 grid for clarity
        for ax, col in zip(axes.flatten(), price_cols):
            ax.plot(df_filtered["Date"], df_filtered[col])
            ax.set_title(col)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")

        plt.tight_layout()
        st.pyplot(fig_ohlc)

        st.caption(
            "These four charts show the **Open**, **High**, **Low** and **Close** price for each day. "
            "Together, they give a more detailed view of the daily trading range."
        )

        st.markdown("### Candlestick chart for the most recent period")

        # I am taking the last N days from the filtered data for the candlestick chart
        sample = df_filtered.tail(sample_days)

        fig_candle = go.Figure(
            data=[
                go.Candlestick(
                    x=sample["Date"],
                    open=sample["Open"],
                    high=sample["High"],
                    low=sample["Low"],
                    close=sample["Close"],
                    name="BTC",
                )
            ]
        )

        fig_candle.update_layout(
            title=f"Bitcoin OHLC – Last {sample_days} Days (Candlestick)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=500,
        )

        # I am stretching the candlestick chart to fill the container width
        st.plotly_chart(fig_candle, width="stretch")

        st.caption(
            "Each candlestick shows one day. The thick body shows where the price opened and closed. "
            "The thin lines (wicks) show the highest and lowest price reached that day."
        )

        st.markdown("### Grouped trend by chosen period")

        freq_code, freq_label = resample_freq
        # I am resampling the closing price to view long-term patterns
        resampled = data["Close"].resample(freq_code).mean()

        fig_res, ax_res = plt.subplots(figsize=(12, 4))
        ax_res.plot(resampled.index, resampled.values, marker="o")
        ax_res.set_title(f"Average {freq_label} Closing Price")
        ax_res.set_ylabel("Price (USD)")
        ax_res.set_xlabel("Date")
        plt.tight_layout()
        st.pyplot(fig_res)

        st.caption(
            "Here I am grouping the data by the period you selected in the sidebar "
            "(year, quarter, or month) and taking the average closing price. "
            "This smooths out daily noise and highlights long-term direction."
        )

    # ======== TAB 3: Volatility view ========
    with tab_volatility:
        st.markdown("### Daily percentage change in closing price")

        fig_pct, ax_pct = plt.subplots(figsize=(14, 4))
        # I am plotting daily percentage changes to visualise volatility
        ax_pct.plot(data_filtered.index, data_filtered["Close_pct_change"])
        ax_pct.set_title("Daily % Change in Bitcoin Closing Price")
        ax_pct.set_ylabel("Percentage Change (%)")
        ax_pct.axhline(0, linestyle="--", alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_pct)

        st.markdown(
            """
Here I am looking at how much the price changed **from one day to the next**, in percentage terms.

- Positive bars show days when the price went up.  
- Negative bars show days when the price went down.  
- Large spikes (up or down) mean very volatile days.

This view helps the user understand risk: how sharply the price could move in a single day.
            """
        )

        st.write("#### Summary of daily % moves for the selected period")
        st.write(data_filtered["Close_pct_change"].describe())

        st.caption(
            "The table above shows key statistics such as the average daily change, "
            "the smallest drop, and the biggest jump. I am using this to describe "
            "how calm or wild the market was during the selected period."
        )

    # ======== TAB 4: About the data ========
    with tab_data:
        st.markdown("### Dataset preview")
        st.dataframe(df.head(10))

        st.write("**Shape (rows, columns):**", df.shape)
        st.write("**Columns:**", df.columns.tolist())

        st.markdown(
            """
**Column meanings in simple terms:**

- `Date` – the trading day  
- `Open` – price at the start of the day (USD per 1 BTC)  
- `High` – highest price reached that day (USD per 1 BTC)  
- `Low` – lowest price reached that day (USD per 1 BTC)  
- `Close` – price at the end of the day (USD per 1 BTC)  
- `Volume` (if present) – total amount of Bitcoin traded that day  

I am using this structured historical data to build a clear, step-by-step view of how Bitcoin behaved between 2013 and 2017.
            """
        )

        st.markdown("### Project summary")
        st.markdown(
            """
In this project, I:

- Loaded and cleaned historical Bitcoin price data (2013–2017)  
- Turned dates into a proper time series so I could work with time-based filters and groupings  
- Built interactive charts to show overall trends, detailed daily prices and candlestick views  
- Measured how much the price moved day-to-day to explain volatility in plain language  
- Grouped prices by **year**, **quarter** and **month** to reveal bigger patterns beyond daily noise  

This app showcases my ability to work with **financial time-series data**, perform **exploratory data analysis (EDA)**,
and build clear, insightful **data visualisations** using Python and Streamlit.
            """
        )

    # ========= Footer / About me =========
    # I am adding a footer so the user knows who built this app and where to find my work
    st.markdown(
        """
<hr>

<div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.95rem;">
  <div>
    <strong>Built by <em>Linda Mthembu</em> – Junior Data Analyst & Aspiring Data Engineer</strong><br>
    This Bitcoin analytics dashboard is powered by historical daily price data from <strong>2013–2017</strong>,
    stored in <code>data/bitcoin_price_training.csv</code>, with all prices shown in <strong>US Dollars (USD)</strong>.
  </div>

  <div style="margin-top:0.5rem;">
    <span>Connect with me:</span>
    <span style="margin-left:8px;">
      <!-- I am using emoji-style icons so they render everywhere without extra libraries -->
      <a href="https://github.com/deereallinda/deerealllinda" target="_blank" style="text-decoration:none; margin-right:16px;">
        <span style="font-size:20px;">🐙</span>
        <span style="margin-left:4px;">GitHub</span>
      </a>
      <a href="https://www.linkedin.com/in/linda-mthembu-66b877270/" target="_blank" style="text-decoration:none;">
        <span style="font-size:20px;">💼</span>
        <span style="margin-left:4px;">LinkedIn</span>
      </a>
    </span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
