**A Time Series Regression Approach to Promotion Optimization **


📌 Overview

This project analyzes daily revenue patterns for a UK-based online retail company using statistical regression and time-series forecasting techniques.
The study aims to:

Quantify the impact of promotions on revenue

Measure weekday vs. weekend purchasing behavior

Build a 90-day SARIMAX forecasting model with exogenous variables

The analysis is built using R, leveraging GLM, SARIMAX, and standard forecasting tools.

**🧠 Key Insights**

🔹 1. Promotion Impact

Promotions increase revenue by ~105% (Gamma GLM)

SARIMAX estimates ~35k GBP additional revenue during promotional periods

Effects are statistically significant (p < 0.01)

🔹 2. Weekend Effect

Revenue drops ~50% on weekends

Indicates B2B purchasing pattern (high weekday activity)

🔹 3. Seasonal Patterns

Strong weekly seasonality (7-day cycle)

Higher volumes during Q4 (Oct–Dec)

Summer dip around July–August

🔹 4. Forecasting Performance

Test RMSE: 21,619

Test MAE: 13,914

Moderate accuracy (maps well to weekly patterns, struggles on extreme spikes)

🛠️ Methods & Models Used
1️⃣ Gamma Generalized Linear Model (GLM)

Used for explaining drivers of revenue

Distribution: Gamma

Link function: Log

Predictors:

is_promotion

is_weekend

month (categorical)

2️⃣ SARIMAX Time Series Model

Used for forecasting future revenue

Auto ARIMA selected: ARIMA(1,0,2)

Seasonal frequency: 7

Exogenous regressors: promotion, weekend

Diagnostics included ACF, PACF, Ljung-Box test

3️⃣ Stationarity Tests

ADF Test: Non-stationary

KPSS Test: Rejects stationarity
→ Seasonal differencing required for forecasting

4️⃣ Evaluation Metrics

RMSE

MAE

MAPE

AIC / AICc

📈 Findings (Summary)

Promotions significantly boost revenue (up to 2x uplift).

Weekends consistently underperform (approx 50% drop).

Revenue is highly seasonal with clear weekly patterns.

SARIMAX is useful for planning but not perfect for extreme spikes.

Combining GLM (interpretability) and SARIMAX (forecasting) gives the best full-picture view.

💼 Business Recommendations

Run promotions on weekdays for maximum ROI

Plan inventory and staffing around Q4 surge

Reduce marketing budget on weekends (low conversion)

Build A/B tests on promotion timing

Integrate additional variables (holidays, competitor campaigns) for next model iteration

📎 How to Run the Code

Clone the repository:

git clone https://github.com/MLAlchemy-cmd/Regression_study


Open the R scripts in RStudio.

Install required packages:

install.packages(c("forecast", "ggplot2", "tseries", "dplyr", "broom", "gridExtra"))


Run the analysis scripts sequentially to reproduce:

Data cleaning

GLM modeling

Stationarity tests

SARIMAX forecasting

Visualization
