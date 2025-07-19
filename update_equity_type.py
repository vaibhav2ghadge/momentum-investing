import pandas as pd
import psycopg2
from psycopg2 import Error
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
POOL_MAX = 30    

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

def execute():

    try:
        # Connect to PostgreSQL database
        pool = create_connection_pool()
        
        # Create a cursor object to execute PostgreSQL queries
        connection = pool.getconn()
        cursor = connection.cursor()
        # Read CSV file
        df = pd.read_csv('/Users/vaibhav.ghadge/Downloads/scrap/EQUITY_L.csv')  # Replace with your CSV file path
        
        # Get all symbols from CSV
        symbols = df['SYMBOL'].tolist()
        print(symbols)
        # Update type to 'stock' where symbols match
        for symbol in symbols:
            update_query = """
                UPDATE ticker_Data 
                SET type = 'EQUITY'
                WHERE ticker = %s
            """
            cursor.execute(update_query, (symbol,))
        
        # Commit the transaction
        connection.commit()
        
        print(f"Updated {cursor.rowcount} rows in stock_prices table")
        
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL or processing data:", error,Error)

    finally:
        # Close database connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection closed")


execute()