import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import statistics
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datetime import datetime
from typing import cast


print("TVS Motor Business Analytics Project")
print()

# 1. Data Collection

ticker = "TVSMOTOR.NS"

data = cast(
    pd.DataFrame,
    yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )
)

if data.empty:
    raise ValueError("No data was downloaded for the selected ticker.")

# Convert multi-level columns into normal column names when needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [
        column[0] if isinstance(column, tuple) else column
        for column in data.columns
    ]

print("Source: Yahoo Finance")
print("Date of collection:", datetime.now().strftime("%d-%m-%Y"))
print("Number of records:", len(data))
print("Variables:", list(data.columns))


# 2. Data Preparation

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())

data = data.drop_duplicates()

for column in ["Open", "High", "Low", "Close", "Volume"]:
    if column not in data.columns:
        raise ValueError(f"Column '{column}' was not found in the downloaded data.")

    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna()


# 3. Feature Creation

data["Daily_Return"] = data["Close"].pct_change() * 100
data["Price_Spread"] = data["High"] - data["Low"]
data["SMA_20"] = data["Close"].rolling(20).mean()
data["SMA_50"] = data["Close"].rolling(50).mean()

data["Previous_Close"] = data["Close"].shift(1)
data["Previous_Return"] = data["Daily_Return"].shift(1)
data["Next_Day_Close"] = data["Close"].shift(-1)

date_index = pd.DatetimeIndex(data.index)
data["Month"] = date_index.month
data["Quarter"] = date_index.quarter

data = data.dropna()

print("\nCleaned data:")
print(data.head())


# 4. Descriptive Statistics

print("\nDescriptive Statistics")

summary = data[
    ["Open", "High", "Low", "Close", "Volume", "Daily_Return"]
].describe()

print(summary)

mean_close = float(data["Close"].mean())
median_close = float(data["Close"].median())
std_close = float(data["Close"].std())
range_close = float(data["Close"].max() - data["Close"].min())

median_volume = float(statistics.median(data["Volume"].tolist()))

print("\nMean closing price:", round(mean_close, 2))
print("Median closing price:", round(median_close, 2))
print("Standard deviation:", round(std_close, 2))
print("Price range:", round(range_close, 2))
print("Median volume:", round(median_volume, 0))

q1 = float(data["Close"].quantile(0.25))
q2 = float(data["Close"].quantile(0.50))
q3 = float(data["Close"].quantile(0.75))

print("25th percentile:", round(q1, 2))
print("50th percentile:", round(q2, 2))
print("75th percentile:", round(q3, 2))

close_skewness = float(skew(data["Close"].to_numpy()))
close_kurtosis = float(kurtosis(data["Close"].to_numpy()))

print("Skewness:", round(close_skewness, 2))
print("Kurtosis:", round(close_kurtosis, 2))


# 5. Frequency Distribution

print("\nPrice Frequency Table")

price_groups = pd.cut(data["Close"], bins=5)
frequency_table = data.groupby(price_groups, observed=False).size()

print(frequency_table)


# 6. Grouping and Aggregation

monthly_summary = data.groupby("Month").agg(
    Average_Close=("Close", "mean"),
    Average_Volume=("Volume", "mean"),
    Average_Return=("Daily_Return", "mean")
).round(2)

print("\nMonthly Summary")
print(monthly_summary)

quarter_summary = data.groupby("Quarter").agg(
    Average_Close=("Close", "mean"),
    Average_Volume=("Volume", "mean")
).round(2)

print("\nQuarterly Summary")
print(quarter_summary)


# 7. Trend Analysis

first_price = float(data["Close"].iloc[0])
last_price = float(data["Close"].iloc[-1])

price_change = last_price - first_price
price_change_percent = (price_change / first_price) * 100

print("\nTrend Analysis")
print("Starting price:", round(first_price, 2))
print("Latest price:", round(last_price, 2))
print("Price change:", round(price_change, 2))
print("Percentage change:", round(price_change_percent, 2), "%")

if price_change_percent > 0:
    print("Overall trend: Upward")
else:
    print("Overall trend: Downward")


# 8. Correlation Analysis

correlation_data = data[
    [
        "Open",
        "High",
        "Low",
        "Volume",
        "Previous_Close",
        "Previous_Return",
        "Next_Day_Close"
    ]
]

print("\nCorrelation Matrix")
print(correlation_data.corr().round(2))

previous_close_correlation = float(
    data[["Previous_Close", "Next_Day_Close"]]
    .corr()
    .iloc[0, 1]
)

print(
    "Correlation between previous day's close and next day's close:",
    round(previous_close_correlation, 3)
)


# 9. Visualizations

plt.figure(figsize=(11, 5))
plt.plot(data.index, data["Close"], label="Closing Price")
plt.plot(data.index, data["SMA_20"], label="20-Day SMA")
plt.plot(data.index, data["SMA_50"], label="50-Day SMA")
plt.title("TVS Motor Closing Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(
    monthly_summary.index.astype(str),
    monthly_summary["Average_Volume"]
)
plt.title("Average Monthly Trading Volume")
plt.xlabel("Month")
plt.ylabel("Average Volume")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.hist(
    data["Daily_Return"],
    bins=30,
    edgecolor="black"
)
plt.axvline(
    float(data["Daily_Return"].mean()),
    linestyle="--",
    label="Average Return"
)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.scatter(
    data["Previous_Close"],
    data["Next_Day_Close"],
    alpha=0.6
)
plt.title("Previous Close vs Next-Day Close")
plt.xlabel("Previous Day Closing Price (INR)")
plt.ylabel("Next Day Closing Price (INR)")
plt.tight_layout()
plt.show()

quarter_data = []
quarter_labels = []

for quarter in sorted(data["Quarter"].unique()):
    values = data.loc[data["Quarter"] == quarter, "Close"]

    if len(values) > 0:
        quarter_data.append(values.to_numpy())
        quarter_labels.append("Q" + str(int(quarter)))

plt.figure(figsize=(8, 5))
plt.boxplot(quarter_data)
plt.title("Closing Price Distribution by Quarter")
plt.xlabel("Quarter")
plt.ylabel("Closing Price (INR)")
plt.xticks(
    range(1, len(quarter_labels) + 1),
    quarter_labels
)
plt.tight_layout()
plt.show()


# 10. Predictive Analysis

X = data[["Previous_Close"]]
y = data["Next_Day_Close"]

split_point = int(len(data) * 0.80)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredictive Analysis")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

intercept = float(model.intercept_)
coefficient = float(model.coef_[0])

print(
    "Regression equation:",
    "Next Day Close =",
    round(intercept, 2),
    "+",
    round(coefficient, 4),
    "* Previous Close"
)


# 11. Model Evaluation

mae = float(mean_absolute_error(y_test, predictions))
mse = float(mean_squared_error(y_test, predictions))
rmse = float(np.sqrt(mse))
r2 = float(r2_score(y_test, predictions))

print("\nModel Evaluation")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R-squared:", round(r2, 4))

results = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions
})

print("\nActual vs Predicted")
print(results.head(10).round(2))

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.to_numpy(), label="Actual")
plt.plot(y_test.index, predictions, label="Predicted")
plt.title("Actual vs Predicted TVS Motor Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price (INR)")
plt.legend()
plt.tight_layout()
plt.show()


# 12. Simple What-If Analysis

latest_close = float(data["Close"].iloc[-1])

scenarios = {
    "5% lower previous close": latest_close * 0.95,
    "Current previous close": latest_close,
    "5% higher previous close": latest_close * 1.05
}

print("\nWhat-If Analysis")

for name, price in scenarios.items():
    scenario_data = pd.DataFrame({
        "Previous_Close": [price]
    })

    predicted_price = float(model.predict(scenario_data)[0])

    print(name, "->", round(predicted_price, 2))

print("\nProject completed.")
