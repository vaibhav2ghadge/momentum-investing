import polars as pl
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
import calendar
import copy

# Suppress Polars' pending deprecation warnings for this script
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Configuration ---
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "stock_data"
DB_USER = "admin"
DB_PASSWORD = "admin"
POOL_MIN = 1
POOL_MAX = 15
RISK_FREE_RATE = 0.065  # Annualized risk-free rate (matches minimum return for simplicity)
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2025, 7, 4)
PORTFOLIO_SIZE = 30
RANK_CUTOFF = 70
MIN_MEDIAN_VOLUME_USD = 10_000_000  # Median volume in USD
MIN_RETURN = 0.065  # 6.5% annual return
MAX_WORKERS = 10
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
INITIAL_CAPITAL = 10_000_000 # Starting capital of 1 crore

# --- Database Connection ---
def create_connection_pool():
    """Creates a threaded connection pool for PostgreSQL."""
    try:
        pool = ThreadedConnectionPool(
            POOL_MIN, POOL_MAX,
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        return pool
    except psycopg2.OperationalError as e:
        print(f"Error creating connection pool: {e}")
        raise

def get_all_tickers(pool):
    """Fetches all unique tickers from the database."""
    conn = pool.getconn()
    try:
        query = "SELECT DISTINCT ticker FROM ticker_data where type is not null;"
        with conn.cursor() as cur:
            cur.execute(query)
            tickers = [row[0] for row in cur.fetchall()]
        return tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []
    finally:
        pool.putconn(conn)

def fetch_stock_data_for_period(pool, tickers, start_date, end_date):
    """Fetches all stock data for a given period."""
    conn = pool.getconn()
    try:
        query = f"""
        SELECT ticker, date, close, volume, circuit
        FROM stock_prices
        WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
        ORDER BY ticker, date;
        """
        df = pl.read_database(query=query, connection=conn, infer_schema_length=None, execute_options={"parameters": [tickers, start_date, end_date]})
        return df
    except Exception as e:
        print(f"Error fetching data for period {start_date} to {end_date}: {e}")
        return pl.DataFrame()
    finally:
        pool.putconn(conn)


def calculate_performance_metrics(monthly_performance, completed_trades):
    """Calculates max drawdown, drawdown days, and top/bottom stock performers."""
    performance_df = pl.DataFrame(monthly_performance).to_pandas()
    performance_df['date'] = pd.to_datetime(performance_df['date'])
    performance_df.set_index('date', inplace=True)
    
    # Max Drawdown Calculation
    rolling_max = performance_df['value'].cummax()
    drawdown = (performance_df['value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Longest Drawdown Duration
    # Identify periods where drawdown is active
    in_drawdown = (drawdown < 0)
    drawdown_start_date = None
    longest_drawdown_duration = timedelta(0)

    for i in range(len(in_drawdown)):
        if in_drawdown.iloc[i] and drawdown_start_date is None:
            drawdown_start_date = performance_df.index[i]
        elif not in_drawdown.iloc[i] and drawdown_start_date is not None:
            duration = performance_df.index[i] - drawdown_start_date
            if duration > longest_drawdown_duration:
                longest_drawdown_duration = duration
            drawdown_start_date = None
    
    # Handle case where drawdown extends to the end of the period
    if drawdown_start_date is not None:
        duration = performance_df.index[-1] - drawdown_start_date
        if duration > longest_drawdown_duration:
            longest_drawdown_duration = duration

    longest_drawdown_months = longest_drawdown_duration.days / 30.44 # Average days in a month

    if completed_trades:
        stock_perf_df = pd.DataFrame(completed_trades)
        stock_perf_df['return'] = pd.to_numeric(stock_perf_df['return'], errors='coerce')
        stock_perf_df.dropna(subset=['return'], inplace=True)
        gainers = stock_perf_df.groupby('ticker')['return'].mean().nlargest(20)
        losers = stock_perf_df.groupby('ticker')['return'].mean().nsmallest(20)
    else:
        gainers, losers = pd.Series(), pd.Series()

    return max_drawdown, longest_drawdown_months, gainers, losers

# --- Reporting Functions ---
def get_color_for_return(ret):
    if pd.isna(ret): return (255, 255, 255)
    capped_ret = max(-0.1, min(0.1, ret))
    if capped_ret >= 0:
        intensity = capped_ret / 0.1
        r, g, b = int(255 - 55 * intensity), 255, int(255 - 55 * intensity)
    else:
        intensity = abs(capped_ret) / 0.1
        r, g, b = 255, int(255 - 55 * intensity), int(255 - 55 * intensity)
    return (r, g, b)

def generate_pdf_report(summary_stats, raw_returns_table, formatted_returns_table, gainers, losers, trade_log, completed_trades):
    pdf = FPDF()
    
    # --- Summary Page ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Performance Report", 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    for key, value in summary_stats.items():
        pdf.cell(80, 10, f"{key}:")
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, str(value), 0, 1)
    
    # --- Monthly/Yearly Returns Table ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Monthly & Yearly Returns", 0, 1, 'C')
    pdf.ln(5)

    page_width = pdf.w - 2 * pdf.l_margin
    num_columns = len(raw_returns_table.columns) + 1
    col_width = page_width / num_columns

    pdf.set_font("Arial", 'B', 8)
    pdf.cell(col_width, 10, "Year", 1)
    for month in raw_returns_table.columns:
        pdf.cell(col_width, 10, month, 1)
    pdf.ln()

    for i, (index, row) in enumerate(raw_returns_table.iterrows()):
        pdf.set_font("Arial", '', 8)
        pdf.cell(col_width, 10, str(index), 1)
        for j, ret in enumerate(row):
            color = get_color_for_return(ret)
            pdf.set_fill_color(color[0], color[1], color[2])
            
            if sum(color) < 382: 
                pdf.set_text_color(255, 255, 255)
            else:
                pdf.set_text_color(0, 0, 0)

            pdf.cell(col_width, 10, formatted_returns_table.iloc[i, j], 1, 0, 'C', fill=True)
            pdf.set_text_color(0, 0, 0)
        pdf.ln()

    

    # --- Top/Bottom Performers ---
    if not gainers.empty:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Top 20 Gainers", 0, 1, 'C')
        pdf.ln(5)
        pdf.set_font("Arial", '', 10)
        for ticker, ret in gainers.items(): pdf.cell(0, 10, f"{ticker}: {ret:.2%}", 0, 1)

    if not losers.empty:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Top 20 Losers", 0, 1, 'C')
        pdf.ln(5)
        pdf.set_font("Arial", '', 10)
        for ticker, ret in losers.items(): pdf.cell(0, 10, f"{ticker}: {ret:.2%}", 0, 1)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Monthly Trade Log", 0, 1, 'C')
    pdf.ln(5)
    for entry in trade_log:
        date_str = entry['date'].strftime('%Y-%m-%d')
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Rebalance Date: {date_str}", 0, 1)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 5, f"Added: {', '.join(entry['added']) if entry['added'] else 'None'}")
        pdf.multi_cell(0, 5, f"Removed: {', '.join(entry['removed']) if entry['removed'] else 'None'}")
        pdf.ln(2)
    
    # --- Individual Trade Performance ---
    if completed_trades:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Individual Trade Performance", 0, 1, 'C')
        pdf.ln(5)

        # Table Headers
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(30, 10, "Ticker", 1)
        pdf.cell(40, 10, "Entry Date", 1)
        pdf.cell(40, 10, "Exit Date", 1)
        pdf.cell(40, 10, "Gain/Loss %", 1)
        pdf.ln()

        # Table Rows
        pdf.set_font("Arial", '', 10)
        for trade in completed_trades:
            pdf.cell(30, 10, trade['ticker'], 1)
            pdf.cell(40, 10, trade['entry_date'].strftime('%Y-%m-%d'), 1)
            pdf.cell(40, 10, trade['exit_date'].strftime('%Y-%m-%d'), 1)
            pdf.cell(40, 10, f"{trade['return']:.2%}", 1)
            pdf.ln()

    pdf.output("performance_report.pdf")
    print("\nPDF report 'performance_report.pdf' generated.")

def generate_returns_charts(monthly_performance):
    if len(monthly_performance) <= 1: return
    performance_df = pl.DataFrame(monthly_performance)
    plt.figure(figsize=(14, 7))
    plt.plot(performance_df['date'], performance_df['value'], linestyle='-')
    plt.title('Growth of 1CR Investment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Rupees)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cumulative_return_over_time.png')
    print("Cumulative return chart 'cumulative_return_over_time.png' generated.")
    plt.close()

# --- Backtesting Engine ---
def get_latest_price_on_or_before_date(df_data, ticker, target_date):
    """Gets the latest closing price for a ticker on or before a target_date from a DataFrame."""
    filtered_df = df_data.filter(
        (pl.col("ticker") == ticker) & (pl.col("date") <= target_date)
    ).sort("date", descending=True)
    if not filtered_df.is_empty():
        return filtered_df.head(1)["close"].item()
    return None

def calculate_metrics_and_rank(df_all_stocks):
    """Calculates metrics for all stocks using Formula 2 (return / volatility) and returns ranked DataFrame."""
    if df_all_stocks.is_empty() or df_all_stocks.height < TRADING_DAYS_PER_YEAR:
        print("return")
        return pl.DataFrame()

    metrics_df = (
        df_all_stocks.sort(["ticker", "date"])
        .group_by("ticker", maintain_order=True)
        .agg([
            pl.col("close").last().alias("current_price"),
            pl.col("close").tail(TRADING_DAYS_PER_YEAR).max().alias("all_time_high"),
            pl.col("close").tail(TRADING_DAYS_PER_YEAR).first().alias("price_one_year_ago"),
            pl.col("close").tail(TRADING_DAYS_PER_YEAR).first().alias("price_one_month_ago"),
            pl.col("close").ewm_mean(span=200, adjust=False).last().alias("ema_200"),
            (pl.col("close") * pl.col("volume")).tail(TRADING_DAYS_PER_YEAR).mean().alias("median_volume"),
            #pl.col("volume").tail(TRADING_DAYS_PER_YEAR).median().alias("median_volume"),
            
            # Calculate daily log returns: ln(Price_t / Price_{t-1})
            (pl.col("close").log() - pl.col("close").shift(1).log()).tail(TRADING_DAYS_PER_YEAR).alias("daily_returns"),
            pl.col("circuit").tail(TRADING_DAYS_PER_YEAR).sum().alias("circuit_count_last_year")
        ])
        .filter(pl.col("daily_returns").list.len() >= TRADING_DAYS_PER_YEAR - 20)
    )

    if metrics_df.is_empty():
        print("return")
        return pl.DataFrame()
    metrics_df = metrics_df.sort("median_volume", descending=True).head(700)
    ranked_df = metrics_df.with_columns([
        # Simple 1Y and 1M returns
        ((pl.col("current_price") / pl.col("price_one_year_ago")) - 1).alias("one_year_return"),
        ((pl.col("current_price") / pl.col("price_one_month_ago")) - 1).alias("one_month_return"),
        # Volatility = stddev of daily log returns
        pl.col("daily_returns").list.std().alias("volatility"),
        # Formula 2: Return / Volatility (using current_price / price_one_year_ago - 1)
        (((pl.col("current_price")/pl.col("price_one_year_ago"))-1)/ pl.col("daily_returns").list.std()).alias("score_return")
        #(((pl.col("current_price") / pl.col("price_one_year_ago")) - 1) / pl.col("daily_returns").list.std()).alias("score_return")
    ]).drop("daily_returns")
    
    
    # Apply filters
    filtered_df = ranked_df.filter(
        (pl.col("current_price") >= 0.75 * pl.col("all_time_high")) &
         (pl.col("current_price") > pl.col("ema_200")) &
         (pl.col("one_year_return") >= MIN_RETURN) &
         #(pl.col("median_volume")> 8000000) &
         #(pl.col("one_month_return") < (0.5 * pl.col("one_year_return"))) &
        # (pl.col("median_volume_usd") >= MIN_MEDIAN_VOLUME_USD) &
         (pl.col("score_return").is_not_nan() & pl.col("score_return").is_finite()) &
         (pl.col("circuit_count_last_year") <= 25)
    )
    #print(len(filtered_df))
    #top_700_df = filtered_df.sort("median_volume", descending=True).head(700)
    #print(f"top {top_700_df.head()['median_volume'][0]}  mediaum {top_700_df[int(len(top_700_df//2))]['median_volume'][0]} last top {top_700_df.tail()['median_volume'][0]} ")
    # Final sort by score_return
    #return top_700_df.sort("score_return", descending=True)
    return filtered_df.sort("score_return", descending=True)



def main():
    pool = create_connection_pool()
    all_tickers = get_all_tickers(pool)
    if not all_tickers: return

    total_capital = INITIAL_CAPITAL
    portfolio_holdings = {}
    # monthly_performance stores the portfolio value at the END of each rebalance period
    monthly_performance = [{'date': (START_DATE - relativedelta(months=1)).date(), 'value': INITIAL_CAPITAL}]
    trade_log, completed_trades = [], []

    current_date = START_DATE
    pbar = tqdm(total=(END_DATE.year - START_DATE.year) * 12 + END_DATE.month - START_DATE.month)

    while current_date <= END_DATE:
        rebalance_month_start = current_date.replace(day=1)
        ranking_date = rebalance_month_start - timedelta(days=1) # Data up to this date for ranking
        next_month_start = rebalance_month_start + relativedelta(months=1)

        # Fetch data for the entire period needed for this iteration
        iter_start_date = rebalance_month_start - relativedelta(years=1) - relativedelta(days=200)
        iter_end_date = rebalance_month_start + timedelta(days=7) # Look into the first week for rebalance day
        all_stocks_data_for_period = fetch_stock_data_for_period(pool, all_tickers, iter_start_date, iter_end_date)
        if all_stocks_data_for_period.is_empty():
            # If no data, append last known value for the current month and skip
            monthly_performance.append({'date': rebalance_month_start.date(), 'value': monthly_performance[-1]['value']})
            trade_log.append({'date': rebalance_month_start.date(), 'added': [], 'removed': []})
            current_date = next_month_start
            pbar.update(1)
            continue

        # Find the first trading day on or after the 1st of the month
        rebalance_day_data = all_stocks_data_for_period.filter(pl.col("date") >= rebalance_month_start)
        if rebalance_day_data.is_empty():
            print(f"\nNo trading data found for the first week of {rebalance_month_start.strftime('%Y-%m')}. Skipping month.")
            # If no trading day, append last known value for the current month and skip
            monthly_performance.append({'date': rebalance_month_start.date(), 'value': monthly_performance[-1]['value']})
            trade_log.append({'date': rebalance_month_start.date(), 'added': [], 'removed': []})
            current_date = next_month_start
            pbar.update(1)
            continue
        
        rebalance_date = rebalance_day_data['date'].min()

        # --- DATA FOR TRANSACTIONS AND VALUATION ---
        # Get prices *exactly* on rebalance_date for actual trades
        transaction_prices_df = all_stocks_data_for_period.filter(pl.col("date") == rebalance_date)
        price_map_today = {row['ticker']: row['close'] for row in transaction_prices_df.to_dicts()}

        # Create a map of latest available prices on or before rebalance_date for all relevant tickers
        price_map_for_valuation = {}
        for ticker in all_tickers: # Consider only tickers that might be in portfolio or ranked
            latest_price = get_latest_price_on_or_before_date(all_stocks_data_for_period, ticker, rebalance_date)
            if latest_price is not None:
                price_map_for_valuation[ticker] = latest_price

        # 1. Calculate current portfolio value at rebalance time
        current_portfolio_value = 0
        if not portfolio_holdings:
            # For the first rebalance, the value is the initial capital.
            # For subsequent months where the portfolio might be empty, it's the last recorded value.
            current_portfolio_value = monthly_performance[-1]['value']
        else:
            for ticker, holding in portfolio_holdings.items():
                price = get_latest_price_on_or_before_date(all_stocks_data_for_period, ticker, rebalance_date)
                if price is not None:
                    current_portfolio_value += holding['shares'] * float(price)
                else:
                    print(f"Warning: No price for currently held {ticker} on {rebalance_date}. Using entry price for valuation.")
                    current_portfolio_value += holding['shares'] * holding['entry_price']

        # 2. Decide new portfolio composition based on T-1 data
        ranking_data = all_stocks_data_for_period.filter(pl.col("date") <= ranking_date)
        ranked_stocks = calculate_metrics_and_rank(ranking_data)
        print(ranked_stocks.height)
        #print(f"top {ranked_stocks.head()[0]['median_volume'][0]} 10 {ranked_stocks[10][0]['median_volume'][0]} 30 {ranked_stocks[30][0]['median_volume'][0]} ")
        # for row in ranked_stocks.iter_rows(named=True):
            
        #     print(row)
        # print(len(ranked_stocks))
        # return
        new_portfolio_tickers = set()
        if not ranked_stocks.is_empty():
            rank_map = {ticker: i + 1 for i, ticker in enumerate(ranked_stocks['ticker'].to_list())}
            previous_portfolio_tickers = set(portfolio_holdings.keys())
            
            stocks_to_keep = {t for t in previous_portfolio_tickers if rank_map.get(t, RANK_CUTOFF + 1) <= RANK_CUTOFF}
            
            num_to_buy = PORTFOLIO_SIZE - len(stocks_to_keep)
            stocks_to_buy = set()
            if num_to_buy > 0:
                for ticker in ranked_stocks['ticker'].to_list():
                    if len(stocks_to_buy) >= num_to_buy: break
                    if ticker not in stocks_to_keep:
                        stocks_to_buy.add(ticker)
            
            new_portfolio_tickers = stocks_to_keep.union(stocks_to_buy)

        # 3. Record completed trades for stocks that are sold
        previous_portfolio_tickers = set(portfolio_holdings.keys())
        stocks_to_sell = previous_portfolio_tickers - new_portfolio_tickers
        for ticker in stocks_to_sell:
            holding = portfolio_holdings[ticker]
            exit_price = price_map_today.get(ticker)
            if exit_price is None:
                exit_price = get_latest_price_on_or_before_date(all_stocks_data_for_period, ticker, rebalance_date)

            if exit_price is not None:
                trade_return = (exit_price - holding['entry_price']) / holding['entry_price'] if holding['entry_price'] > 0 else 0
                completed_trades.append({'ticker': ticker, 'return': trade_return, 'entry_date': holding['entry_date'], 'exit_date': rebalance_date})
            else:
                print(f"Warning: No exit price for {ticker} on {rebalance_date}. Assuming 0 return for this trade.")
                completed_trades.append({'ticker': ticker, 'return': 0, 'entry_date': holding['entry_date'], 'exit_date': rebalance_date})

        # 4. Allocate new portfolio
        old_holdings = portfolio_holdings.copy()
        portfolio_holdings = {}
        cash_leftover = 0.0

        if new_portfolio_tickers:
            investment_per_stock = current_portfolio_value / len(new_portfolio_tickers)
            for ticker in new_portfolio_tickers:
                price = price_map_today.get(ticker)
                if price is not None and price > 0:
                    shares = investment_per_stock / float(price)
                    entry_price = old_holdings[ticker]['entry_price'] if ticker in old_holdings else price
                    entry_date = old_holdings[ticker]['entry_date'] if ticker in old_holdings else rebalance_date
                    portfolio_holdings[ticker] = {'shares': shares, 'entry_price': entry_price, 'entry_date': entry_date}
                else:
                    print(f"Warning: Could not buy {ticker} on {rebalance_date} due to no valid price. Holding cash instead.")
                    cash_leftover += investment_per_stock
        else: # If no stocks to hold, all capital becomes cash
            cash_leftover = current_portfolio_value

        # 5. LOGGING AND PERFORMANCE TRACKING
        added_stocks = new_portfolio_tickers - previous_portfolio_tickers
        removed_stocks = previous_portfolio_tickers - new_portfolio_tickers
        trade_log.append({'date': rebalance_date, 'added': list(added_stocks), 'removed': list(removed_stocks)})

        # The value at the end of the rebalance is the sum of new holdings plus any cash leftover
        current_month_end_value = sum(
            h['shares'] * float(price_map_today.get(t, h['entry_price'])) for t, h in portfolio_holdings.items()
        ) + cash_leftover

        # Append the end-of-month value to monthly_performance
        # print(type(rebalance_date))
        # print(type(current_date))
        monthly_performance.append({'date': current_date.date()-relativedelta(months=1), 'value': current_month_end_value})
        
        # Calculate monthly return for the month that just ended
        # Compare current month's end value with previous month's end value
        monthly_return = (monthly_performance[-1]['value'] / monthly_performance[-2]['value']) - 1 if len(monthly_performance) > 1 and monthly_performance[-2]['value'] > 0 else 0
        print(f"{(current_date.date()-relativedelta(months=1)).strftime('%Y-%m-%d')},{monthly_return:.2%}")
        #print(f"\nRebalanced on {rebalance_date.strftime('%Y-%m-%d')}. Portfolio value: Rs. {current_month_end_value:,.2f}. Monthly Return: {monthly_return:.2%}")

        current_date = next_month_start
        pbar.update(1)

    pbar.close()
    
    # --- FINAL REPORTING ---
    if len(monthly_performance) > 1:
        final_capital = monthly_performance[-1]['value']
        total_months = len(monthly_performance) - 1
        returns = pd.Series([p['value'] for p in monthly_performance]).pct_change().dropna().to_list()

        cumulative_return = (final_capital / INITIAL_CAPITAL) - 1
        annualized_return = (1 + cumulative_return) ** (12 / total_months) - 1 if total_months > 0 else 0
        annualized_volatility = np.std(returns) * np.sqrt(12)
        sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility if annualized_volatility > 0 else 0

        max_drawdown, longest_drawdown_months, gainers, losers = calculate_performance_metrics(monthly_performance, completed_trades)

        summary_stats = {
            "Period": f"{START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}",
            "Initial Capital": f"Rs. {INITIAL_CAPITAL:,.2f}",
            "Final Capital": f"Rs. {final_capital:,.2f}",
            "Cumulative Return": f"{cumulative_return:.2%}",
            "Annualized Return (CAGR)": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Longest Drawdown (Months)": f"~{longest_drawdown_months:.1f}"
        }

        pd_performance_df = pl.DataFrame(monthly_performance).to_pandas()
        pd_performance_df['date'] = pd.to_datetime(pd_performance_df['date'])
        pd_performance_df.set_index('date', inplace=True)
        
        # Shift the returns to correctly align with the month they represent
        monthly_returns_for_table = pd_performance_df['value'].resample('M').last().pct_change().dropna()
        monthly_returns_for_table.index = monthly_returns_for_table.index.to_period('M').to_timestamp()

        returns_pivot = monthly_returns_for_table.to_frame(name='return').pivot_table(
            values='return', 
            index=monthly_returns_for_table.index.year, 
            columns=monthly_returns_for_table.index.month
        )
        returns_pivot.columns = [calendar.month_abbr[c] for c in returns_pivot.columns]
        
        # Calculate yearly returns correctly from the monthly data
        yearly_returns = returns_pivot.apply(lambda row: (1 + row.fillna(0)).prod() - 1, axis=1)
        returns_pivot['Yearly'] = yearly_returns
        
        formatted_pivot = returns_pivot.applymap(lambda x: f"{x:.2%}" if not pd.isna(x) else "-")

        generate_pdf_report(summary_stats, returns_pivot, formatted_pivot, gainers, losers, trade_log, completed_trades)
        generate_returns_charts(monthly_performance)

if __name__ == "__main__":
    main()