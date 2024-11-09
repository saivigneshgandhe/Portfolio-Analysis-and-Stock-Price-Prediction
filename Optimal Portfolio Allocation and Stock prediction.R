# Portfolio Analysis and Stock Price Prediction
# Author: Gandhe Sai Vignesh
# Last Updated: 2024-11-09
# Description: This script performs portfolio optimization, Monte Carlo simulation,
#              and machine learning-based stock price prediction for Indian stocks.

# --- Library Imports ---
# Core financial libraries
library(PerformanceAnalytics)  # For financial calculations
library(quantmod)              # For fetching stock data
library(fPortfolio)           # For portfolio optimization
library(timeSeries)           # For time series operations

# Data manipulation libraries
library(dplyr)                # For data manipulation
library(tidyr)                # For data reshaping
library(readxl)               # For Excel file operations
library(writexl)              # For Excel file operations

# Machine learning and visualization
library(randomForest)         # For predictive modeling
library(ggplot2)              # For visualization
library(forecast)             # For time series forecasting

# --- Global Variables ---
# Define stock portfolio
symbols <- c(
  "HDFCBANK.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS", 
  "ITC.NS", "PERSISTENT.NS", "FEDERALBNK.NS", "TATAPOWER.NS",
  "PHOENIXLTD.NS", "DATAPATTNS.NS", "KEI.NS", "MAZDOCK.NS",
  "JBMA.NS", "KPITTECH.NS"
)


portfolio_stk <- lapply(symbols,function(X){
  getSymbols(X,from=start_date,to=end_date,auto.assign = FALSE)
})


portfolio_stk <- na.omit(merge(Ad(HDFCBANK.NS), Ad(RELIANCE.NS), Ad(INFY.NS),Ad(TCS.NS),Ad(ITC.NS),Ad(PERSISTENT.NS),Ad(FEDERALBNK.NS),Ad(TATAPOWER.NS),Ad(PHOENIXLTD.NS),Ad(DATAPATTNS.NS),Ad(KEI.NS),Ad(MAZDOCK.NS),Ad(JBMA.NS),Ad(KPITTECH.NS)))

portfolio_stk <- na.omit(portfolio_stk)
head(portfolio_stk)


portfolio_stk_df <- as.timeSeries(portfolio_stk)
portfolio_stk_df
head(portfolio_stk_df)


portfolio_stk_ret <- Return.calculate(portfolio_stk_df,method='discrete')
# Creating equal weights for 14 stocks
weights <- rep(1/14, 14)  # This creates a vector of 14 equal weights

portfolio_stk_ret_pf <- Return.portfolio(portfolio_stk_ret, 
                                         weights = weights,
                                         geometric = FALSE)
portfolio_stk_ret <- na.omit(portfolio_stk_ret)
head(portfolio_stk_ret_pf)
mean(portfolio_stk_ret_pf)
var_covarince <- cov(portfolio_stk_ret_pf)

# Extracting only the 'mean' and 'Sigma' elements
min_var_return <- as.numeric(getTargetReturn(min_var_portfolio)["mean"])
min_var_risk <- as.numeric(getTargetRisk(min_var_portfolio)["Sigma"])
tangency_return <- as.numeric(getTargetReturn(optimum_portfolio)["mean"])
tangency_risk <- as.numeric(getTargetRisk(optimum_portfolio)["Sigma"])

efficient_pf <- portfolioFrontier(portfolio_stk_ret,`setRiskFreeRate<-`(portfolioSpec(),0.07/252), constraints = 'longOnly')


# Plotting efficient frontier
plot(efficient_pf, c(1, 2, 3, 5, 7))

#Points for minimum variance and tangency portfolios
points(min_var_risk, min_var_return, col = "red", pch = 19, cex = 1.5)
points(tangency_risk, tangency_return, col = "blue", pch = 19, cex = 1.5)
legend("topright", legend = c("Min Variance", "Tangency Portfolio"),
       col = c("red", "blue"), pch = 19)



min_var_portfolio <- minvariancePortfolio(portfolio_stk_ret,portfolioSpec(),constraints='longonly')
wts_min_portfolio <- getWeights(min_var_portfolio)                              
wts_min_portfolio

optimum_portfolio <- tangencyPortfolio(portfolio_stk_ret,`setRiskFreeRate<-`(portfolioSpec(), .07/246),constraints='longonly')
optimum_portfolio

# Extracting portfolio weights
weights <- getWeights(optimum_portfolio)

# Calculate portfolio returns
portfolio_returns <- Return.portfolio(portfolio_stk_ret, weights = weights)

# Calculates Sharpe Ratio
mean_return <- mean(portfolio_returns)
sd_return <- sd(portfolio_returns)
risk_free_rate <- 0.07/252  # Daily risk-free rate

sharpe_ratio <- (mean_return - risk_free_rate) / sd_return
sharpe_ratio

# Calculates VaR and CVaR at 95% confidence level
VaR_95 <- quantile(portfolio_returns, 0.05)
CVaR_95 <- mean(portfolio_returns[portfolio_returns <= VaR_95])

cat("Value at Risk (95%):", VaR_95, "\n")
cat("Conditional Value at Risk (95%):", CVaR_95, "\n")

# Saves portfolio weights to an Excel file
weights_df <- data.frame(Symbol = symbols, Weights = weights)
write_xlsx(weights_df, "Optimal_Portfolio_Weights.xlsx")

# Exports portfolio returns
portfolio_returns_df <- data.frame(Date = index(portfolio_returns), Return = portfolio_returns)
write_xlsx(portfolio_returns_df, "Portfolio_Returns.xlsx")


###MonteCarlo Simulation#######


# Defining number of simulations and time horizon (e.g., 252 days = 1 year)
num_simulations <- 10000
time_horizon <- 252

# Calculating mean returns and covariance matrix from historical data
mean_returns <- colMeans(portfolio_stk_ret)
cov_matrix <- cov(portfolio_stk_ret)

# Initializing matrix to store simulated portfolio returns
simulated_portfolio_returns <- numeric(num_simulations)

# Performing Monte Carlo simulation
for (i in 1:num_simulations) {
  # Generate random returns for each stock using multivariate normal distribution
  simulated_returns <- mvrnorm(n = time_horizon, mu = mean_returns, Sigma = cov_matrix)
  
  # Convert simulated returns to an xts object with a date index
  date_index <- seq.Date(from = start_date, by = "day", length.out = time_horizon)
  simulated_returns_xts <- xts(simulated_returns, order.by = date_index)
  
  # Calculate portfolio returns for the simulated time horizon
  portfolio_returns <- Return.portfolio(simulated_returns_xts, weights = weights, geometric = FALSE)
  
  # Calculate total return for the portfolio over the time horizon
  simulated_portfolio_returns[i] <- prod(1 + portfolio_returns) - 1
}

# Summarize the simulation results
mean_simulated_return <- mean(simulated_portfolio_returns)
sd_simulated_return <- sd(simulated_portfolio_returns)

# Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) at 95% confidence level
VaR_95 <- quantile(simulated_portfolio_returns, 0.05)
CVaR_95 <- mean(simulated_portfolio_returns[simulated_portfolio_returns <= VaR_95])


# Plot the distribution of simulated returns
hist(simulated_portfolio_returns, breaks = 50, main = "Monte Carlo Simulation of Portfolio Returns",
     xlab = "Simulated Portfolio Returns", col = "lightblue", border = "white")
abline(v = VaR_95, col = "red", lwd = 2, lty = 2)
abline(v = CVaR_95, col = "darkred", lwd = 2, lty = 2)
legend("topright", legend = c("VaR (95%)", "CVaR (95%)"),
       col = c("red", "darkred"), lwd = 2, lty = 2)


###Integrating Machine Learning for Stock returns prediction###

# Define stock symbols and time period
symbols <- c("HDFCBANK.NS","RELIANCE.NS","INFY.NS","TCS.NS","ITC.NS","PERSISTENT.NS","FEDERALBNK.NS","TATAPOWER.NS","PHOENIXLTD.NS","DATAPATTNS.NS","KEI.NS","MAZDOCK.NS","JBMA.NS","KPITTECH.NS")
start_date <- as.Date("2015-01-01")
end_date <- Sys.Date()

# Retrieve stock data and calculate daily returns
portfolio_data <- lapply(symbols, function(sym) {
  stock_data <- getSymbols(sym, from = start_date, to = end_date, auto.assign = FALSE)
  daily_returns <- dailyReturn(Cl(stock_data))
  colnames(daily_returns) <- sym
  return(daily_returns)
})
portfolio_returns <- na.omit(do.call(merge, portfolio_data))

# Prepare Data for Model Training
prepare_features <- function(data) {
  # Convert to numeric if necessary (data may be a time series or xts object)
  data <- as.numeric(data)
  
  # Create lagged features
  lagged_data <- data.frame(
    Return = data,
    Lag1 = lag(data, 1),
    Lag2 = lag(data, 2),
    Lag3 = lag(data, 3)
  )
  
  # Remove rows with NA values due to lagging
  lagged_data <- na.omit(lagged_data)
  
  return(lagged_data)
}

# Train and predict returns for each stock using Random Forest
predicted_returns <- lapply(names(portfolio_returns), function(sym) {
  # Ensure 'portfolio_returns' is a numeric vector and prepare features
  stock_returns <- portfolio_returns[, sym]
  features <- prepare_features(stock_returns)
  
  # Split data into training and test sets (80-20 split)
  train_size <- floor(0.8 * nrow(features))
  train_data <- features[1:train_size, ]
  test_data <- features[(train_size + 1):nrow(features), ]
  
  # Train the Random Forest model
  rf_model <- randomForest(Return ~ ., data = train_data, ntree = 100)
  
  # Predict returns for the test set
  predictions <- predict(rf_model, newdata = test_data)
  
  # Convert predictions to a data frame and name the column by the stock symbol
  pred_df <- data.frame(Date = index(test_data), Predicted_Return = predictions)
  colnames(pred_df) <- c("Date", sym)
  
  return(pred_df)
})

# Merge all predicted returns into one data frame by Date
predicted_returns_df <- Reduce(function(x, y) merge(x, y, by = "Date", all = TRUE), predicted_returns)

# Update Date column to start from tomorrow and increment for each prediction
start_date <- Sys.Date() + 1
date_seq <- seq(start_date, by = "day", length.out = nrow(predicted_returns_df))
predicted_returns_df$Date <- date_seq

# View the final merged predicted returns data frame
head(predicted_returns_df)

# Function to calculate future prices from returns
calculate_future_prices <- function(last_price, predicted_returns) {
  # Convert returns to price multipliers (1 + return)
  price_multipliers <- 1 + predicted_returns
  
  # Calculate cumulative product of multipliers
  cumulative_multipliers <- cumprod(price_multipliers)
  
  # Calculate future prices
  future_prices <- last_price * cumulative_multipliers
  
  return(future_prices)
}

# Get the last known prices for each stock
get_last_prices <- function(symbols) {
  last_prices <- sapply(symbols, function(sym) {
    stock_data <- getSymbols(sym, auto.assign = FALSE)
    last_price <- as.numeric(tail(Cl(stock_data), 1))
    return(last_price)
  })
  return(last_prices)
}

# Plot predictions for all stocks
plot_predictions <- function(symbols, predicted_returns_df, last_prices) {
  # Create sequence of future dates
  future_dates <- seq(Sys.Date() + 1, by = "day", length.out = nrow(predicted_returns_df))
  
  # Initialize list to store predicted prices for each stock
  predicted_prices <- list()
  
  # Calculate predicted prices for each stock
  for (sym in symbols) {
    predicted_returns <- predicted_returns_df[[sym]]
    last_price <- last_prices[sym]
    predicted_prices[[sym]] <- calculate_future_prices(last_price, predicted_returns)
  }
  
  # Convert to data frame for plotting
  plot_data <- data.frame(Date = future_dates)
  for (sym in symbols) {
    plot_data[[sym]] <- predicted_prices[[sym]]
  }
  
  # Reshape data for ggplot
  plot_data_long <- pivot_longer(plot_data, 
                                 cols = -Date,
                                 names_to = "Stock",
                                 values_to = "Price")
  
  # Create the plot
  p <- ggplot(plot_data_long, aes(x = Date, y = Price, color = Stock)) +
    geom_line() +
    theme_minimal() +
    labs(title = "Predicted Stock Prices",
         x = "Date",
         y = "Price (INR)",
         color = "Stock") +
    theme(legend.position = "right",
          plot.title = element_text(hjust = 0.5, size = 16),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.text = element_text(size = 8)) +
    scale_y_continuous(labels = scales::comma) +
    scale_x_date(date_breaks = "1 week", date_labels = "%Y-%m-%d")
  
  return(p)
}

# Function to save predictions and create summary
create_prediction_summary <- function(symbols, predicted_returns_df, last_prices) {
  # Initialize summary data frame
  summary_df <- data.frame(
    Symbol = symbols,
    Last_Price = last_prices,
    stringsAsFactors = FALSE
  )
  
  # Calculate predicted prices for each stock
  for (sym in symbols) {
    predicted_returns <- predicted_returns_df[[sym]]
    last_price <- last_prices[sym]
    predicted_prices <- calculate_future_prices(last_price, predicted_returns)
    
    # Add summary statistics
    summary_df$Pred_Price_30d[summary_df$Symbol == sym] <- tail(predicted_prices, 1)
    summary_df$Expected_Return[summary_df$Symbol == sym] <- 
      (tail(predicted_prices, 1) - last_price) / last_price * 100
  }
  
  # Add predicted direction (Up/Down)
  summary_df$Direction <- ifelse(summary_df$Expected_Return > 0, "Up", "Down")
  
  return(summary_df)
}

# Main execution
# Get the last known prices
last_prices <- get_last_prices(symbols)

# Create the plot
p <- plot_predictions(symbols, predicted_returns_df, last_prices)

# Display the plot
print(p)

# Create and display summary
summary_df <- create_prediction_summary(symbols, predicted_returns_df, last_prices)
print(summary_df)

# Save the plot
ggsave("stock_predictions.pdf", p, width = 12, height = 8)

# Save the summary to CSV
write.csv(summary_df, "prediction_summary.csv", row.names = FALSE)

# Create individual plots for each stock
for (sym in symbols) {
  # Subset data for single stock
  stock_data <- data.frame(
    Date = seq(Sys.Date() + 1, by = "day", length.out = nrow(predicted_returns_df)),
    Price = calculate_future_prices(last_prices[sym], predicted_returns_df[[sym]])
  )
  
  # Create individual plot
  p_individual <- ggplot(stock_data, aes(x = Date, y = Price)) +
    geom_line(color = "blue") +
    theme_minimal() +
    labs(title = paste("Predicted Prices for", sym),
         x = "Date",
         y = "Price (INR)") +
    theme(plot.title = element_text(hjust = 0.5, size = 16),
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_y_continuous(labels = scales::comma) +
    scale_x_date(date_breaks = "1 week", date_labels = "%Y-%m-%d")
  
  # Save individual plot
  ggsave(paste0("predictions_", gsub(".NS", "", sym), ".pdf"), 
         p_individual, 
         width = 8, 
         height = 6)
}



