import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import newton
import requests
from datetime import datetime, timedelta

# =============================================
# Helper function moved to global scope for robustness
# =============================================
def find_default(options, keywords):
    """Intelligently finds the best default index for a selectbox."""
    lower_options = [str(opt).lower() for opt in options]
    for keyword in keywords:
        # First, look for an exact match
        if keyword in lower_options:
            return lower_options.index(keyword)
        # Then, look for a partial match
        for i, option in enumerate(lower_options):
            if keyword in option:
                return i
    return 0 # Default to the first column if no match is found

# =============================================
# 1. Core Data Processing Functions
# =============================================

@st.cache_data
def load_raw_data(uploaded_files):
    """
    Loads and concatenates CSVs. It now warns and skips any empty or unreadable files
    instead of stopping the entire process.
    """
    all_trades = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            # Only append the dataframe if it's not empty
            if not df.empty:
                all_trades.append(df)
            else:
                st.warning(f"Skipping file '{file.name}' because it is empty.")
        except pd.errors.EmptyDataError:
            # This specific error happens when the file is empty or has no columns
            st.warning(f"Skipping file '{file.name}' because it has no columns or data to parse.")
            continue # Move to the next file
        except Exception as e:
            st.warning(f"Skipping file '{file.name}' due to an unexpected error: {e}")
            continue # Move to the next file
            
    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True)

def standardize_data(df, mapping):
    """Creates a standardized DataFrame based on user's column mapping."""
    try:
        standard_df = pd.DataFrame({
            'symbol': df[mapping['symbol']],
            'trade_date': pd.to_datetime(df[mapping['date']]),
            'trade_type': df[mapping['type']],
            'quantity': pd.to_numeric(df[mapping['quantity']]),
            'price': pd.to_numeric(df[mapping['price']]),
            'currency': 'USD'
        })
        standard_df['quantity'] = standard_df['quantity'].abs()
        return standard_df
    except Exception as e:
        st.error(f"An error occurred during data standardization: {e}")
        return pd.DataFrame()

# (All other analysis functions like get_split_data, etc. remain the same)
@st.cache_data
def get_split_data(_symbols):
    splits_data = {}
    for symbol in _symbols:
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            if not splits.empty:
                splits_data[symbol] = splits
        except Exception:
            pass
    return splits_data

def adjust_trades_for_splits(trades_df, splits_data):
    adj_trades_df = trades_df.copy()
    adj_trades_df['adj_quantity'] = adj_trades_df['quantity']
    adj_trades_df['adj_price'] = adj_trades_df['price']
    for index, trade in adj_trades_df.iterrows():
        symbol = trade['symbol']
        trade_date = trade['trade_date']
        if symbol in splits_data:
            symbol_splits = splits_data[symbol]
            for split_date, ratio in symbol_splits[symbol_splits.index > trade_date].items():
                adj_trades_df.loc[index, 'adj_quantity'] *= ratio
                adj_trades_df.loc[index, 'adj_price'] /= ratio
    adj_trades_df['adj_cashflow'] = adj_trades_df['adj_quantity'] * adj_trades_df['adj_price']
    return adj_trades_df

@st.cache_data
def get_historical_data(_symbols, start_date, end_date):
    start = (start_date - timedelta(days=5)).strftime('%Y-%m-%d')
    end = (end_date + timedelta(days=5)).strftime('%Y-%m-%d')
    symbols_to_fetch = list(_symbols) + ['INR=X', 'SGD=X']
    try:
        data = yf.download(symbols_to_fetch, start=start, end=end, auto_adjust=False, progress=False)
        prices = data['Adj Close'].dropna(how='all').ffill()
        forex = prices[['INR=X', 'SGD=X']].rename(columns={'INR=X': 'USD_INR', 'SGD=X': 'USD_SGD'})
        stock_prices = prices.drop(columns=['INR=X', 'SGD=X'])
        return stock_prices, forex
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def calculate_daily_portfolio_value(adj_trades_df, stock_prices, forex_data):
    date_range = pd.date_range(start=adj_trades_df['trade_date'].min(), end=stock_prices.index.max())
    symbols = adj_trades_df['symbol'].unique()
    daily_holdings = pd.DataFrame(index=date_range, columns=symbols).fillna(0)
    for symbol in symbols:
        symbol_trades = adj_trades_df[adj_trades_df['symbol'] == symbol].sort_values('trade_date')
        qty_changes = symbol_trades.set_index('trade_date')['adj_quantity']
        qty_changes[symbol_trades['trade_type'].str.contains('sell', case=False)] *= -1
        daily_holdings[symbol] = qty_changes.cumsum().reindex(date_range, method='ffill').fillna(0)
    portfolio_value_usd = (daily_holdings * stock_prices).sum(axis=1)
    portfolio_df = pd.DataFrame(portfolio_value_usd, columns=['Value_USD'])
    portfolio_df = portfolio_df.merge(forex_data, left_index=True, right_index=True, how='left').ffill()
    portfolio_df['Value_INR'] = portfolio_df['Value_USD'] * portfolio_df['USD_INR']
    portfolio_df['Value_SGD'] = portfolio_df['Value_USD'] * portfolio_df['USD_SGD']
    return portfolio_df[['Value_USD', 'Value_INR', 'Value_SGD']]

def xnpv(rate, values, dates):
    if rate <= -1.0: return float('inf')
    return sum([val / (1 + rate)**((date - dates[0]).days / 365.0) for val, date in zip(values, dates)])

def calculate_xirr(values, dates):
    try:
        return newton(lambda r: xnpv(r, values, dates), 0.1)
    except (RuntimeError, TypeError):
        return float('nan')

def compute_all_xirr(adj_trades_df, latest_prices):
    xirr_results = {}
    symbols = adj_trades_df['symbol'].unique()
    today = datetime.now()
    for symbol in symbols:
        symbol_trades = adj_trades_df[adj_trades_df['symbol'] == symbol].sort_values('trade_date')
        dates = symbol_trades['trade_date'].to_list()
        cashflows = symbol_trades.apply(
            lambda row: -row['adj_cashflow'] if row['trade_type'].str.contains('buy', case=False) else row['adj_cashflow'], 
            axis=1
        ).to_list()
        buy_qty = symbol_trades[symbol_trades['trade_type'].str.contains('buy', case=False)]['adj_quantity'].sum()
        sell_qty = symbol_trades[symbol_trades['trade_type'].str.contains('sell', case=False)]['adj_quantity'].sum()
        current_qty = buy_qty - sell_qty
        if current_qty > 0 and symbol in latest_prices and pd.notna(latest_prices[symbol]):
            final_value = current_qty * latest_prices[symbol]
            cashflows.append(final_value)
            dates.append(today)
        if len(cashflows) > 1:
             xirr_results[symbol] = calculate_xirr(cashflows, dates)
        else:
             xirr_results[symbol] = float('nan')
    return xirr_results

@st.cache_data(ttl=3600)
def get_news(_symbols, api_key):
    if not api_key: return {}
    all_news = {}
    for symbol in _symbols:
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}&pageSize=3&sortBy=publishedAt"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            all_news[symbol] = articles
        except requests.exceptions.RequestException:
            pass
    return all_news

# =============================================
# 4. Streamlit User Interface
# =============================================

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("📈 Stock Portfolio Analyzer")

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload Trade CSV Files", type=['csv'], accept_multiple_files=True
    )
    st.header("2. API Key (Optional)")
    news_api_key = st.text_input("NewsAPI.org API Key", type="password")

    if uploaded_files:
        raw_data = load_raw_data(uploaded_files)
        if not raw_data.empty:
            st.header("3. Map Your Columns")
            st.info("Select the columns from your file that correspond to each field.")
            available_cols = raw_data.columns.tolist()

            col_mapping = {
                'symbol': st.selectbox("Symbol", available_cols, index=find_default(available_cols, ['symbol'])),
                'date': st.selectbox("Date", available_cols, index=find_default(available_cols, ['date/time', 'date'])),
                'type': st.selectbox("Trade Type", available_cols, index=find_default(available_cols, ['datadiscriminator', 'type'])),
                'quantity': st.selectbox("Quantity", available_cols, index=find_default(available_cols, ['quantity'])),
                'price': st.selectbox("Price", available_cols, index=find_default(available_cols, ['price', 't._price'])),
            }

            if st.button("🚀 Analyze Portfolio", use_container_width=True):
                st.session_state.raw_data = raw_data
                st.session_state.col_mapping = col_mapping
                st.session_state.news_api_key = news_api_key
                st.session_state.start_analysis = True
                st.session_state.analysis_done = False
        else:
            st.warning("All uploaded files were skipped. Please upload at least one valid CSV file.")


if 'start_analysis' in st.session_state and st.session_state.start_analysis:
    with st.spinner("Standardizing data and running analysis..."):
        raw_data = st.session_state.raw_data
        col_mapping = st.session_state.col_mapping
        standard_trades = standardize_data(raw_data, col_mapping)
        
        if not standard_trades.empty:
            unique_symbols = standard_trades['symbol'].unique()
            start_date = standard_trades['trade_date'].min()
            end_date = datetime.now()
            splits = get_split_data(unique_symbols)
            adj_trades = adjust_trades_for_splits(standard_trades, splits)
            stock_prices, forex_data = get_historical_data(unique_symbols, start_date, end_date)
            
            if not stock_prices.empty:
                daily_value = calculate_daily_portfolio_value(adj_trades, stock_prices, forex_data)
                latest_prices = stock_prices.iloc[-1]
                xirr_results = compute_all_xirr(adj_trades, latest_prices)
                xirr_df = pd.DataFrame(list(xirr_results.items()), columns=['Symbol', 'XIRR'])
                xirr_df['XIRR (%)'] = xirr_df['XIRR'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")

                st.session_state.daily_value = daily_value
                st.session_state.xirr_df = xirr_df
                st.session_state.standard_trades = standard_trades
                st.session_state.adj_trades = adj_trades
                st.session_state.analysis_done = True
            else:
                 st.error("Could not retrieve market data to complete analysis.")
        st.session_state.start_analysis = False

if st.session_state.analysis_done:
    st.success("Analysis complete! ✅")
    st.header("Portfolio Value Over Time")
    st.line_chart(st.session_state.daily_value)
    st.header("Holding Performance (XIRR)")
    st.dataframe(st.session_state.xirr_df[['Symbol', 'XIRR (%)']], use_container_width=True)
    with st.expander("View Detailed Trade Data"):
        st.subheader("Mapped & Standardized Data")
        st.dataframe(st.session_state.standard_trades, use_container_width=True)
        st.subheader("Split-Adjusted Data")
        st.dataframe(st.session_state.adj_trades, use_container_width=True)
    if st.session_state.news_api_key:
        st.header("📰 Latest News")
        unique_symbols = st.session_state.standard_trades['symbol'].unique()
        news_data = get_news(unique_symbols, st.session_state.news_api_key)
        if not news_data: st.info("No news found or API key is invalid.")
        else:
            for symbol, articles in news_data.items():
                if articles:
                    st.subheader(f"News for {symbol}")
                    for article in articles:
                        st.markdown(f"- **[{article['title']}]({article['url']})** _{article['source']['name']}_")
else:
    if uploaded_files:
        st.markdown("---")
        st.subheader("Raw Data Preview")
        st.dataframe(load_raw_data(uploaded_files).head(10))
    else:
        st.info("Upload your trade files in the sidebar to begin analysis.")
