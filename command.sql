CREATE TABLE stock_prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    primary key (ticker, date)
);

-- Create indexes for faster queries
CREATE INDEX idx_ticker ON stock_prices (ticker);
CREATE INDEX idx_ticker_date ON stock_prices (ticker, date);