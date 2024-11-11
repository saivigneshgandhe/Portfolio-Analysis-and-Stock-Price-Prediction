# Portfolio Analysis and Stock Price Prediction
By Gandhe Sai Vignesh
Last Updated: 2024-11-09

## Project Overview
This project implements a comprehensive financial analysis toolkit for Indian stocks, combining portfolio optimization, risk analysis, Monte Carlo simulation, and machine learning-based price prediction. The analysis focuses on a portfolio of 14 selected Indian stocks from various sectors.

## Features

### 1. Portfolio Optimization
- Efficient frontier calculation
- Minimum variance portfolio identification
- Tangency portfolio optimization
- Risk-free rate consideration (7% annual)
- Long-only constraints implementation

### 2. Risk Analysis
- Sharpe ratio calculation
- Value at Risk (VaR) assessment
- Conditional Value at Risk (CVaR) computation
- Variance-covariance analysis

### 3. Monte Carlo Simulation
- 10,000 simulation scenarios
- 252-day (1 year) time horizon
- Distribution analysis of potential returns
- Visual representation of simulation results

### 4. Machine Learning Price Prediction
- Random Forest model implementation
- Feature engineering with lagged returns
- 80-20 train-test split
- Individual stock prediction analysis
- Consolidated prediction visualization

## File Structure

### Input Files
- `portfolio_analysis.R` - Main R script containing all analysis code
- Historical stock price data (automatically fetched)

### Output Files
1. **Visualization Files**
   - `stock_predictions.pdf` - Combined predictions for all stocks
   - `predictions_[STOCKNAME].pdf` - Individual stock prediction charts
   - Efficient frontier plot
   - Monte Carlo simulation distribution

2. **Data Files**
   - `Optimal_Portfolio_Weights.xlsx` - Optimized portfolio weights
   - `Portfolio_Returns.xlsx` - Historical portfolio returns
   - `prediction_summary.csv` - Summary of stock predictions

## Stock Portfolio
The analysis covers the following Indian stocks:
- HDFC Bank (HDFCBANK.NS)
- Reliance Industries (RELIANCE.NS)
- Infosys (INFY.NS)
- TCS (TCS.NS)
- ITC (ITC.NS)
- Persistent Systems (PERSISTENT.NS)
- Federal Bank (FEDERALBNK.NS)
- Tata Power (TATAPOWER.NS)
- Phoenix Ltd (PHOENIXLTD.NS)
- Data Patterns (DATAPATTNS.NS)
- KEI Industries (KEI.NS)
- Mazagon Dock (MAZDOCK.NS)
- JBM Auto (JBMA.NS)
- KPIT Technologies (KPITTECH.NS)

## Technical Requirements

### R Dependencies
```R
# Core Financial Analysis
library(PerformanceAnalytics)
library(quantmod)
library(fPortfolio)
library(timeSeries)

# Data Manipulation
library(dplyr)
library(tidyr)
library(readxl)
library(writexl)

# Machine Learning & Visualization
library(randomForest)
library(ggplot2)
library(forecast)
library(MASS)
```

## Installation & Usage

1. Install R and required packages:
   ```R
   install.packages(c("PerformanceAnalytics", "quantmod", "fPortfolio", 
                     "timeSeries", "dplyr", "tidyr", "readxl", "writexl",
                     "randomForest", "ggplot2", "forecast", "MASS"))
   ```

2. Run the analysis:
   ```R
   source("portfolio_analysis.R")
   ```

## Output Interpretation

### 1. Portfolio Optimization
- The efficient frontier plot shows the risk-return tradeoff
- Red dot: Minimum variance portfolio
- Blue dot: Tangency portfolio
- Optimal weights are saved in `Optimal_Portfolio_Weights.xlsx`

### 2. Monte Carlo Simulation
The histogram shows:
- Distribution of potential portfolio returns
- VaR at 95% confidence level (red line)
- CVaR at 95% confidence level (dark red line)

### 3. Stock Predictions
- Individual stock PDFs show predicted price movements
- Combined visualization in `stock_predictions.pdf`
- Summary statistics in `prediction_summary.csv` including:
  - Last known price
  - 30-day price prediction
  - Expected return percentage
  - Predicted direction (Up/Down)

## Time Period
- Historical data: 2015-01-01 to present
- Prediction horizon: Next 30 days

## Notes and Limitations
1. Predictions are based on historical data and should not be used as the sole basis for investment decisions
2. The model assumes relatively stable market conditions
3. Transaction costs and taxes are not considered
4. Past performance does not guarantee future results

## Author
Gandhe Sai Vignesh

## Last Update
November 9, 2024

## License
This project is for educational and research purposes only. Not financial advice.
