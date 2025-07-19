import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from pathlib import Path

from psycopg2.pool import ThreadedConnectionPool
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
CSV_DIR = "/Users/vaibhav.ghadge/projects/backtest/eod2/src/eod2_data/daily/"  # Replace with the path to your CSV files
DB_HOST = "localhost"           # Docker container host
DB_PORT = "5432"               # Docker container port
DB_NAME = "stock_data"         # Database name
DB_USER = "admin"           # Database user (e.g., 'postgres' or 'admin')
DB_PASSWORD = "admin"  # Database password

POOL_MIN = 1                    # Minimum connections in pool
POOL_MAX = 30                   # Maximum connections (match max_workers)

# Create connection pool
def create_connection_pool():
    try:
        pool = ThreadedConnectionPool(
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
        return pool
    except psycopg2.OperationalError as e:
        print(f"Connection pool error: {e}")
        print(f"Failed to connect to {DB_HOST}:{DB_PORT} with user {DB_USER}")
        raise
    except Exception as e:
        print(f"Unexpected error creating pool: {e}")
        raise


# Create stock_prices table and convert to hypertable
def create_stock_table(pool):
    conn = pool.getconn()
    try:
        create_table_query = """
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,2) NOT NULL,
            high DECIMAL(10,2) NOT NULL,
            low DECIMAL(10,2) NOT NULL,
            close DECIMAL(10,2) NOT NULL,
            volume BIGINT NOT NULL,
            total_trades BIGINT,
            qty_per_trade DECIMAL(10,2) ,
            dlv_qty BIGINT ,
            PRIMARY KEY (ticker, date)
        );
        """
        create_hypertable_query = """
        SELECT create_hypertable('stock_prices', 'date', if_not_exists => TRUE);
        """
        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_prices (ticker, date);
        """
        with conn.cursor() as cur:
            cur.execute(create_table_query)
            cur.execute(create_hypertable_query)
            cur.execute(create_index_query)
        conn.commit()
        print("Table 'stock_prices' created and configured as hypertable")
    except Exception as e:
        print(f"Error creating table or hypertable: {e}")
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)



# Insert data into TimescaleDB
def insert_data(conn, df, ticker):
    insert_query = """
    INSERT INTO stock_prices (ticker, date, open, high, low, close, volume, total_trades, qty_per_trade, dlv_qty)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, date) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(insert_query, (
                    ticker,
                    row['Date'],
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    row['TOTAL_TRADES'],
                    row['QTY_PER_TRADE'],
                    row['DLV_QTY']
                ))
        conn.commit()
       # print(f"Inserted data for ticker {ticker} from file {ticker}.csv")
    except Exception as e:
        print(f"Error inserting data for {ticker}: {e} \n {row}")
        conn.rollback()

# Process a single CSV file
def process_single_csv(csv_file, pool):
    ticker = csv_file.stem.upper()  # Extract ticker from filename (e.g., 'AAPL' from 'AAPL.csv')
   # print(f"Processing {csv_file} for ticker {ticker}")
    
    # Get a connection from the pool
    conn = pool.getconn()
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        # Validate columns
        expected_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'TOTAL_TRADES', 'QTY_PER_TRADE', 'DLV_QTY'}
        if not expected_columns.issubset(df.columns):
            print(f"Skipping {csv_file}: Missing required columns")
            return f"Skipped {csv_file}: Missing required columns"
        # Ensure Date is in correct format (YYYY-MM-DD)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['TOTAL_TRADES'] = df['TOTAL_TRADES'].fillna(0).astype(int)
        df['QTY_PER_TRADE'] = df['QTY_PER_TRADE'].fillna(0.0)
        df['DLV_QTY'] = df['DLV_QTY'].fillna(0).astype(int)
        insert_data(conn, df, ticker)
        #return f"Successfully processed {csv_file} for ticker {ticker}"
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return f"Error processing {csv_file}: {e}"
    finally:
        pool.putconn(conn)  # Return connection to pool
            

# Process CSV files in parallel
def process_csv_files(csv_dir, pool):
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory {csv_dir} does not exist")

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the directory")
        return

    # Process up to 10 files concurrently
    max_workers = min(10, len(csv_files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_csv, csv_file, pool): csv_file for csv_file in csv_files}
        for future in as_completed(future_to_file):
            csv_file = future_to_file[future]
            try:
                result = future.result()
                #print(result)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

            
def main():
    # Create connection pool
    pool = create_connection_pool()
    
    # Create table and hypertable
    create_stock_table(pool)
    
    # Process all CSV files
    process_csv_files(CSV_DIR, pool)
  
    print("Connection pool closed")

if __name__ == "__main__":
    main()