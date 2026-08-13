

## Overview

This project analyzes Walmart weekly sales data to identify factors associated with sales performance and builds machine learning models to predict weekly sales.

The analysis focuses on:

- Store-level differences in weekly sales
- Seasonal patterns
- Holiday effects
- Economic factors such as CPI and unemployment
- Environmental factors such as temperature and fuel price
- Outlier sales weeks
- Linear Regression as a baseline model
- Random Forest Regression for nonlinear prediction
- Feature importance analysis
- A custom `holiday_leadup` feature

## Dataset

The dataset contains **6,435 weekly observations** across **45 Walmart stores**.

### Columns

| Column | Description |
|---|---|
| `Store` | Walmart store number |
| `Date` | Week start date |
| `Weekly_Sales` | Weekly sales for the store |
| `Holiday_Flag` | Whether the week is marked as a holiday week |
| `Temperature` | Regional air temperature |
| `Fuel_Price` | Regional fuel price |
| `CPI` | Consumer Price Index |
| `Unemployment` | Regional unemployment rate |

The dataset covers:

- **Start:** February 5, 2010
- **End:** October 26, 2012
- **Stores:** 45
- **Observations:** 6,435
- **Missing values:** None

