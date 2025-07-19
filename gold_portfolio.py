import pandas as pd
import numpy as np
from fpdf import FPDF
import calendar
from fpdf import FPDF
import calendar

def load_and_clean_data(filepath):
    """
    Loads return data from a CSV file, cleans it, and prepares it for analysis.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with a datetime index and numeric returns,
                      or None if the file is not found.
    """
    try:
        # Use skipfooter to ignore non-data lines at the end of the file.
        # The python engine is required for skipfooter.
        df = pd.read_csv(filepath, skipfooter=5, engine='python')
        # Convert 'return' column from '1.44%' string to 0.0144 float
        df['return'] = df['return'].str.replace('%', '', regex=False).astype(float) / 100
        # Use format='mixed' to handle multiple date formats automatically.
        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        # Resample to the start of the month to ensure consistent dates
        df = df.resample('MS').first()
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")
        return None

def get_color_for_return(ret):
    if pd.isna(ret):
        return (255, 255, 255)  # White for no data
    # Cap returns at +/- 10% for color scaling
    capped_ret = max(-0.1, min(0.1, ret))
    if capped_ret >= 0:
        # Green scale for profits
        intensity = capped_ret / 0.1
        r, g, b = int(200 - 200 * intensity), 255, int(200 - 200 * intensity)
    else:
        # Red scale for losses
        intensity = abs(capped_ret) / 0.1
        r, g, b = 255, int(200 - 200 * intensity), int(200 - 200 * intensity)
    return (r, g, b)

def generate_pdf_report(summary_stats, returns_pivot, allocation_df):
    """
    Generates a PDF report with performance metrics, returns grid, and allocation table.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Tactical Allocation Strategy Report", 0, 1, 'C')
    pdf.ln(10)

    # --- Summary Stats ---
    pdf.set_font("Arial", 'B', 12)
    for key, value in summary_stats.items():
        pdf.cell(95, 10, f"{key}:", border=1)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, str(value), border=1, ln=1)
        pdf.set_font("Arial", 'B', 12)
    pdf.ln(10)

    # --- Monthly & Yearly Returns Table ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Monthly & Yearly Returns", 0, 1, 'C')
    pdf.ln(5)

    page_width = pdf.w - 2 * pdf.l_margin
    num_columns = len(returns_pivot.columns) + 1
    col_width = page_width / num_columns

    pdf.set_font("Arial", 'B', 8)
    pdf.cell(col_width, 10, "Year", 1)
    for month in returns_pivot.columns:
        pdf.cell(col_width, 10, month, 1)
    pdf.ln()

    for i, (index, row) in enumerate(returns_pivot.iterrows()):
        pdf.set_font("Arial", '', 8)
        pdf.cell(col_width, 10, str(index), 1)
        for j, ret in enumerate(row.drop('Yearly')):
            color = get_color_for_return(ret)
            pdf.set_fill_color(color[0], color[1], color[2])
            pdf.cell(col_width, 10, f"{ret:.2%}" if not pd.isna(ret) else "-", 1, 0, 'C', fill=True)
        # Yearly return
        yearly_ret = row['Yearly']
        color = get_color_for_return(yearly_ret)
        pdf.set_fill_color(color[0], color[1], color[2])
        pdf.cell(col_width, 10, f"{yearly_ret:.2%}", 1, 0, 'C', fill=True)
        pdf.ln()
    pdf.ln(10)

    # --- Monthly Allocation Table ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Monthly Strategy Allocation", 0, 1, 'C')
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 8)
    pdf.cell(col_width, 10, "Year", 1)
    for month in allocation_df.columns:
        pdf.cell(col_width, 10, month, 1)
    pdf.ln()

    for index, row in allocation_df.iterrows():
        pdf.set_font("Arial", '', 8)
        pdf.cell(col_width, 10, str(index), 1)
        for allocation in row:
            pdf.cell(col_width, 10, str(allocation) if pd.notna(allocation) else "-", 1, 0, 'C')
        pdf.ln()

    pdf.output("tactical_allocation_report.pdf")
    print("\nPDF report 'tactical_allocation_report.pdf' generated.")

def get_color_for_return(ret):
    if pd.isna(ret):
        return (255, 255, 255)  # White for no data
    # Cap returns at +/- 10% for color scaling
    capped_ret = max(-0.1, min(0.1, ret))
    if capped_ret >= 0:
        # Green scale for profits
        intensity = capped_ret / 0.1
        r, g, b = int(200 - 200 * intensity), 255, int(200 - 200 * intensity)
    else:
        # Red scale for losses
        intensity = abs(capped_ret) / 0.1
        r, g, b = 255, int(200 - 200 * intensity), int(200 - 200 * intensity)
    return (r, g, b)

def generate_pdf_report(summary_stats, strategy_returns, gold_returns, portfolio_returns, allocation_df):
    """
    Generates a detailed PDF report with performance metrics and color-coded tables.
    """
    pdf = FPDF()
    # --- Page 1: Summary Dashboard ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Tactical Allocation Strategy Report", 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    for key, value in summary_stats.items():
        pdf.cell(95, 10, f"{key}:", border=1)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, str(value), border=1, ln=1)
        pdf.set_font("Arial", 'B', 12)
    pdf.ln(10)

    # --- Page 2: Strategy Returns ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Strategy Monthly & Yearly Returns", 0, 1, 'C')
    pdf.ln(5)
    generate_returns_table(pdf, strategy_returns)

    # --- Page 3: Gold Returns ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Gold Monthly & Yearly Returns", 0, 1, 'C')
    pdf.ln(5)
    generate_returns_table(pdf, gold_returns)

    # --- Page 4: Portfolio Returns ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Portfolio Monthly & Yearly Returns", 0, 1, 'C')
    pdf.ln(5)
    generate_returns_table(pdf, portfolio_returns)

    # --- Page 5: Allocation Table ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Monthly Strategy Allocation", 0, 1, 'C')
    pdf.ln(5)
    generate_allocation_table(pdf, allocation_df)

    pdf.output("tactical_allocation_report.pdf")
    print("\nPDF report 'tactical_allocation_report.pdf' generated.")

def generate_returns_table(pdf, returns_pivot):
    page_width = pdf.w - 2 * pdf.l_margin
    # Add 1 for the 'Year' column to get the correct total number of columns
    num_columns = len(returns_pivot.columns) + 1
    col_width = page_width / num_columns

    pdf.set_font("Arial", 'B', 8)
    pdf.cell(col_width, 10, "Year", 1)
    for month in returns_pivot.columns.drop('Yearly'):
        pdf.cell(col_width, 10, month, 1)
    pdf.cell(col_width, 10, "Yearly", 1)
    pdf.ln()

    for i, (index, row) in enumerate(returns_pivot.iterrows()):
        pdf.set_font("Arial", '', 8)
        pdf.cell(col_width, 10, str(index), 1)
        for j, ret in enumerate(row.drop('Yearly')):
            color = get_color_for_return(ret)
            pdf.set_fill_color(color[0], color[1], color[2])
            pdf.cell(col_width, 10, f"{ret:.2%}" if not pd.isna(ret) else "-", 1, 0, 'C', fill=True)
        yearly_ret = row['Yearly']
        color = get_color_for_return(yearly_ret)
        pdf.set_fill_color(color[0], color[1], color[2])
        pdf.cell(col_width, 10, f"{yearly_ret:.2%}", 1, 0, 'C', fill=True)
        pdf.ln()

def generate_allocation_table(pdf, allocation_df):
    page_width = pdf.w - 2 * pdf.l_margin
    num_columns = len(allocation_df.columns) + 1
    col_width = page_width / num_columns

    pdf.set_font("Arial", 'B', 8)
    pdf.cell(col_width, 10, "Year", 1)
    for month in allocation_df.columns:
        pdf.cell(col_width, 10, month, 1)
    pdf.ln()

    for index, row in allocation_df.iterrows():
        pdf.set_font("Arial", '', 8)
        pdf.cell(col_width, 10, str(index), 1)
        for allocation in row:
            if pd.notna(allocation):
                if allocation == 'gold':
                    pdf.set_fill_color(255, 215, 0)  # Gold
                else:
                    pdf.set_fill_color(144, 238, 144)  # Light Green
                pdf.cell(col_width, 10, str(allocation.capitalize()), 1, 0, 'C', fill=True)
            else:
                pdf.cell(col_width, 10, "-", 1, 0, 'C')
        pdf.ln()

def run_tactical_allocation_strategy():
    """
    Runs the tactical allocation strategy between a stock portfolio and gold.
    """
    # 1. Load and clean the data
    portfolio_df = load_and_clean_data('portfolio_return.csv')
    gold_df = load_and_clean_data('gold_return.csv')

    if portfolio_df is None or gold_df is None:
        print("Exiting due to missing data files.")
        return

    # 2. Combine data into a single DataFrame
    combined_df = pd.merge(
        portfolio_df,
        gold_df,
        on='date',
        how='inner',
        suffixes=('_portfolio', '_gold')
    )

    # 3. Calculate 12-month trailing returns
    combined_df['trailing_12m_portfolio'] = (1 + combined_df['return_portfolio']).rolling(window=6).apply(np.prod, raw=True) - 1
    combined_df['trailing_12m_gold'] = (1 + combined_df['return_gold']).rolling(window=6).apply(np.prod, raw=True) - 1

    strategy_df = combined_df.dropna(subset=['trailing_12m_portfolio', 'trailing_12m_gold']).copy()

    # 4. Implement the switching logic
    strategy_df['allocation'] = np.where(
        strategy_df['trailing_12m_gold'] > strategy_df['trailing_12m_portfolio'],
        'gold',
        'portfolio'
    )
    strategy_df['allocation'] = strategy_df['allocation'].shift(1)
    strategy_df.dropna(subset=['allocation'], inplace=True)

    # 5. Calculate strategy returns
    strategy_df['strategy_return'] = np.where(
        strategy_df['allocation'] == 'gold',
        strategy_df['return_gold'],
        strategy_df['return_portfolio']
    )

    # 6. Calculate performance metrics
    start_date = strategy_df.index.min().strftime('%Y-%m-%d')
    end_date = strategy_df.index.max().strftime('%Y-%m-%d')
    switch_count = (strategy_df['allocation'] != strategy_df['allocation'].shift(1)).sum()
    total_months = len(strategy_df)
    num_years = total_months / 12
    cumulative_return = (1 + strategy_df['strategy_return']).prod()
    cagr = (cumulative_return ** (1 / num_years)) - 1

    # Max Drawdown
    strategy_df['cumulative_return'] = (1 + strategy_df['strategy_return']).cumprod()
    strategy_df['rolling_max'] = strategy_df['cumulative_return'].cummax()
    strategy_df['drawdown'] = (strategy_df['cumulative_return'] - strategy_df['rolling_max']) / strategy_df['rolling_max']
    max_drawdown = strategy_df['drawdown'].min()

    # Longest Drawdown Duration
    in_drawdown = strategy_df['drawdown'] < 0
    drawdown_periods = (in_drawdown != in_drawdown.shift(1)).cumsum()
    drawdown_lengths = drawdown_periods[in_drawdown].value_counts()
    longest_drawdown_months = drawdown_lengths.max() if not drawdown_lengths.empty else 0

    # Sharpe Ratio (assuming risk-free rate of 0)
    annualized_volatility = strategy_df['strategy_return'].std() * np.sqrt(12)
    sharpe_ratio = cagr / annualized_volatility if annualized_volatility > 0 else 0

    summary_stats = {
        "Period": f"{start_date} to {end_date}",
        "Strategy CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Longest Drawdown (Months)": longest_drawdown_months,
        "Total Switches": switch_count
    }

    # 7. Prepare data for PDF report
    def create_pivot_table(df, value_col):
        pivot = df.pivot_table(values=value_col, index=df.index.year, columns=df.index.month)
        pivot = pivot.reindex(columns=range(1, 13))
        pivot.columns = [calendar.month_abbr[i] for i in range(1, 13)]
        pivot['Yearly'] = pivot.apply(lambda row: (1 + row.fillna(0)).prod() - 1, axis=1)
        return pivot

    strategy_returns_pivot = create_pivot_table(strategy_df, 'strategy_return')
    gold_returns_pivot = create_pivot_table(strategy_df, 'return_gold')
    portfolio_returns_pivot = create_pivot_table(strategy_df, 'return_portfolio')

    allocation_pivot = strategy_df.pivot_table(
        values='allocation',
        index=strategy_df.index.year,
        columns=strategy_df.index.month,
        aggfunc='first'
    )
    allocation_pivot = allocation_pivot.reindex(columns=range(1, 13))
    allocation_pivot.columns = [calendar.month_abbr[i] for i in range(1, 13)]

    # 8. Generate PDF
    generate_pdf_report(summary_stats, strategy_returns_pivot, gold_returns_pivot, portfolio_returns_pivot, allocation_pivot)

    # Print yearly returns to console as before
    print("--- Tactical Allocation Strategy Results ---")
    print(f"Strategy CAGR: {cagr:.2%}")
    print(f"Total Switches: {switch_count}")
    print("\n--- Yearly Returns ---")
    yearly_returns = strategy_returns_pivot['Yearly']
    print(yearly_returns.to_string(header=True, float_format='{:.2%}'.format))

if __name__ == "__main__":
    run_tactical_allocation_strategy()
