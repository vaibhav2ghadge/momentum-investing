
---

# Momentum Investing Backtest


## Overview

This repository contains Python code for backtesting a momentum investing strategy applied to all-cap stocks (small, mid, and large-cap). The strategy ranks stocks based on their **Sharpe ratio** to identify high-momentum stocks and evaluates portfolio performance through historical data. The backtest includes portfolio construction, rebalancing, and performance metrics such as returns, volatility, and Sharpe ratio.

### Features
- **Momentum Strategy**: Ranks stocks based on historical momentum using the Sharpe ratio.
- **All-Cap Coverage**: Includes small, mid, and large-cap stocks for diversified exposure.
- **Backtesting Framework**: Simulates portfolio performance over historical data.
- **Performance Metrics**: Calculates returns, volatility, Sharpe ratio, and drawdowns.
- **Customizable Parameters**: Adjust lookback periods, rebalancing frequency, and portfolio size.

### Performance
![Yearly Performance](https://github.com/vaibhav2ghadge/momentum-investing/blob/main/Momo-yearly-stat.png?raw=true)

![Metrics](https://github.com/vaibhav2ghadge/momentum-investing/blob/main/Momo-stat.png)

## Prerequisites

To run the code, ensure you have the following installed:
- Python 3.8+
- Required Python libraries:
  ```bash
  pip install pandas numpy yfinance matplotlib scipy
  ```
- Access to historical stock data (e.g., via `yfinance` or another data provider like Yahoo Finance, Alpha Vantage, or Quandl).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/momentum-investing-backtest.git
   cd momentum-investing-backtest
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data**:
   - The code uses `yfinance` to fetch historical stock data by default. Ensure you have an internet connection.
   - Alternatively, provide your own CSV files with historical stock data in the `data/` directory.

## Usage

1. **Configure Parameters**:
   - Edit `config.py` to set:
     - Lookback period for momentum calculation (e.g., 6 months).
     - Rebalancing frequency (e.g., monthly, quarterly).
     - Number of stocks in the portfolio.
     - Start and end dates for the backtest.

2. **Run the Backtest**:
   ```bash
   python momo.py
   ```

3. **Output**:
   - Results are saved in the `results/` directory, including:
     - Portfolio performance metrics (returns, Sharpe ratio, max drawdown).
     - Visualizations (e.g., portfolio value over time, drawdown plot).
     - CSV files with portfolio holdings and weights.


## Example

To backtest a momentum strategy with a 6-month lookback period, monthly rebalancing, and a portfolio of 20 stocks:
```python
# In config.py
LOOKBACK_PERIOD = 126  # 6 months (assuming 21 trading days/month)
REBALANCE_FREQ = '1M'  # Monthly
PORTFOLIO_SIZE = 20
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'
```

Run:
```bash
python src/momo.py
```

View results in `results/` (e.g., `portfolio_performance.csv`, `equity_curve.png`).

## Methodology

1. **Data Collection**: Fetch historical price data for all-cap stocks using `yfinance` or a custom data source.
2. **Momentum Calculation**: Compute returns and volatility over the lookback period to calculate the Sharpe ratio for each stock.
3. **Ranking**: Rank stocks by Sharpe ratio and select the top N stocks for the portfolio.
4. **Portfolio Construction**: Allocate equal weights to selected stocks.
5. **Rebalancing**: Rebalance the portfolio periodically (e.g., monthly) based on updated rankings.
6. **Performance Evaluation**: Calculate portfolio returns, volatility, Sharpe ratio, and maximum drawdown.

## Limitations
- **Data Quality**: Relies on the accuracy of external data sources like Yahoo Finance.
- **Transaction Costs**: The current model does not account for transaction costs or slippage, which may impact real-world performance.
- **Market Assumptions**: Assumes historical patterns will hold; actual results may vary.
- **Universe of Stocks**: Limited to stocks available in the data source; ensure the dataset covers desired all-cap stocks.
