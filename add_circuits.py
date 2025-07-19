import polars as pl
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2 import Error
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import numpy as np
from psycopg2.pool import ThreadedConnectionPool
from psycopg2 import Error
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connection pool configuration (replace with your actual values)
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "stock_data"
DB_USER = "admin"
DB_PASSWORD = "admin"
POOL_MIN = 1
POOL_MAX = 2

# Circuit breaker thresholds
CIRCUIT_THRESHOLDS = {4.99, 5.00, 9.99, 10.00, 19.99, 20.00}

def create_connection_pool():
    global connection_pool
    try:
        connection_pool = ThreadedConnectionPool(
            POOL_MIN,
            POOL_MAX,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5
        )
        print(f"Created connection pool for {DB_HOST}:{DB_PORT} with user {DB_USER}")
        
    except psycopg2.OperationalError as e:
        print(f"Connection pool error: {e}")
        print(f"Failed to connect to {DB_HOST}:{DB_PORT} with user {DB_USER}")
        raise
    except Exception as e:
        print(f"Unexpected error creating pool: {e}")
        raise

def process_ticker(ticker):
    """Process a single ticker: fetch data, calculate circuits, and update database directly."""
    connection = None
    cursor = None
    try:
    
        # Get connection from shared pool
        connection = connection_pool.getconn()
        cursor = connection.cursor()

        # Fetch data for the ticker for the last 10 years
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=20*365)
        query = """
            SELECT date, low, high, close, circuit
            FROM stock_prices
            WHERE ticker = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """
        df = pl.read_database(query, connection,infer_schema_length=None, execute_options={"parameters": [ticker, start_date, end_date]}).sort("date")

        if df.is_empty():
            logging.info(f"No data found for ticker {ticker}")
            return ticker, 0

    
        # Cast price columns to float64 to avoid decimal.Decimal issues
        df = df.with_columns(
            pl.col("low").cast(pl.Float64, strict=False).alias("low"),
            pl.col("high").cast(pl.Float64, strict=False).alias("high"),
            pl.col("close").cast(pl.Float64, strict=False).alias("close"),
            pl.col("circuit").cast(pl.Int32, strict=False).fill_null(0).alias("circuit")
        )

        # Validate data: log rows with invalid price values
        invalid_rows = df.filter(
            pl.col("low").is_null() | 
            pl.col("high").is_null() | 
            pl.col("close").is_null() | 
            pl.col("low").is_nan() | 
            pl.col("high").is_nan() | 
            pl.col("close").is_nan()
        )
        if not invalid_rows.is_empty():
            logging.warning(f"Ticker {ticker} - Invalid rows (null/NaN): {invalid_rows.to_dicts()}")

        # Calculate previous close and percentage change
        df = df.with_columns(
            pl.col("close").shift(1).cast(pl.Float64, strict=False).alias("prev_close")
        ).with_columns(
            pl.when(
                pl.col("prev_close").is_not_null() & 
                pl.col("close").is_not_null() & 
                (pl.col("prev_close") != 0) &
                pl.col("close").is_not_nan() &
                pl.col("prev_close").is_not_nan()
            )
            .then(
                pl.col("close").sub(pl.col("prev_close"))
                .truediv(pl.col("prev_close"))
                .mul(100)
                .abs()
                .round(2) # Round for accurate comparison
            )
            .otherwise(None)
            .cast(pl.Float64, strict=False)
            .alias("pct_change")
        )


        # Check for circuit breaker conditions separately to avoid bitand error
        # 1. Percentage change within 1e-10 of thresholds
        df = df.with_columns(
            pl.col("pct_change")
            .is_in(CIRCUIT_THRESHOLDS)
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("is_pct_circuit")
        )
        # # 2. close == high or close == low
        # df = df.with_columns(
        #     pl.when(
        #         (pl.col("close").eq(pl.col("high")) | pl.col("close").eq(pl.col("low"))) &
        #         pl.col("low").is_not_null() & 
        #         pl.col("high").is_not_null() & 
        #         pl.col("close").is_not_null() &
        #         pl.col("low").is_not_nan() & 
        #         pl.col("high").is_not_nan() & 
        #         pl.col("close").is_not_nan()
        #     )
        #     .then(True)
        #     .otherwise(False)
        #     .alias("is_price_circuit")
        # )
        # Combine conditions
        df = df.with_columns(
            pl.when(
                # pl.col("is_pct_circuit") | pl.col("is_price_circuit")
                pl.col("is_pct_circuit")
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("new_circuit")
        )

        # Filter rows where circuit needs updating
        df = df.filter(pl.col("new_circuit") != pl.col("circuit"))

        # Prepare batch update
        update_data = df.select(["date", "new_circuit"]).to_dicts()
        if not update_data:
            logging.info(f"No circuit updates needed for ticker {ticker}")
            return ticker, 0

        # Perform direct batch update
        update_query = """
            UPDATE stock_prices
            SET circuit = %s
            WHERE ticker = %s AND date = %s
        """
        cursor.executemany(
            update_query,
            [(row["new_circuit"], ticker, row["date"]) for row in update_data]
        )
        rows_updated = cursor.rowcount

        connection.commit()
        logging.info(f"Updated {rows_updated} rows for ticker {ticker}")
        return ticker, rows_updated

   
    except (Exception, Error) as error:
        if "connection pool is closed" in str(error).lower():
            return ticker, 0
        logging.error(f"Error processing ticker {ticker}: {error}")
        return ticker, 0

    finally:
        if cursor:
            cursor.close()
        if connection:
                connection_pool.putconn(connection)
        

def main():
    global connection_pool
    try:
        # Initialize shared connection pool
        create_connection_pool()

        # Get all unique tickers
        
        connection = connection_pool.getconn()
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM ticker_data where type is not null")
        tickers = [row[0] for row in cursor.fetchall()]
        cursor.close()
        connection_pool.putconn(connection)

        # Process tickers in parallel using ThreadPoolExecutor
        max_workers = min(POOL_MAX, len(tickers))  # Limit threads to pool size or number of tickers
        logging.info(f"Processing {len(tickers)} tickers using {max_workers} threads")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
            results = []
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Ticker {ticker} generated an exception: {e}")
                    results.append((ticker, 0))

        # Summarize results
        total_updated = sum(count for _, count in results)
        logging.info(f"Total rows updated across all tickers: {total_updated}")

    except (Exception, Error) as error:
        logging.error(f"Error in main process: {error}")
       


if __name__ == "__main__":
    main()