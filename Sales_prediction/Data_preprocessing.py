import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the datasets
data_sales = pd.read_csv('./data/sale_products_new.csv')
data_returns = pd.read_csv('./data/sale_return_products.csv')

print(data_sales.head(5))
print(data_returns.head(5))

#check if there are any null values
print(data_sales.isna().sum())
print(data_returns.isna().sum())

# Convert sales_date to datetime
data_sales['sales_date'] = pd.to_datetime(data_sales['sales_date'])
data_returns['sales_date'] = pd.to_datetime(data_returns['sales_date'])

# Extract year and month for both datasets
data_sales['year'] = data_sales['sales_date'].dt.year
data_sales['month'] = data_sales['sales_date'].dt.month

data_returns['year'] = data_returns['sales_date'].dt.year
data_returns['month'] = data_returns['sales_date'].dt.month


# Aggregate sales data to get total sales per month per type_name and product_id
monthly_sales = data_sales.groupby(['year', 'month', 'type_name', 'product_id'])['total'].sum().reset_index()
monthly_sales.to_csv('./data/monthly_sales.csv')

# Aggregate return data to get total returns per month per product_id
monthly_returns = data_returns.groupby(['year', 'month', 'product_id'])['total'].sum().reset_index()
monthly_returns.rename(columns={'total': 'return_total'}, inplace=True)
monthly_returns.to_csv('./data/monthly_returns.csv')

# Merge sales and returns data
merged_data = pd.merge(monthly_sales, monthly_returns, on=['year', 'month', 'product_id'], how='left')
merged_data['return_total'] = merged_data['return_total'].fillna(0)
merged_data['net_total'] = merged_data['total'] - merged_data['return_total']

# Save the processed data to a CSV file for future use
merged_data.to_csv('./data/processed_sales_data.csv', index=False)

# Pivot to create a time-series format
pivot_data = merged_data.pivot_table(index=['year', 'month'], columns='type_name', values='net_total').fillna(0)

# Ensure the index is a datetime index
pivot_data.index = pd.to_datetime(pivot_data.index.map(lambda x: f"{x[0]}-{x[1]}-01"))
pivot_data.to_csv('./data/pivot_data.csv')

# Decompose the time series for a selected product type
product_type = pivot_data.columns[0]
product_data = pivot_data[product_type]

# Perform decomposition
decomposition = seasonal_decompose(product_data, model='additive', period=12)

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(product_data, label='Original Data')
plt.title('Original Time Series')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.title('Trend Component')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.title('Seasonality Component')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.title('Residual Component')
plt.legend()

plt.tight_layout()
plt.show()

# Aggregate overall sales across all product types for decomposition
overall_sales = pivot_data.sum(axis=1)

# Perform decomposition on overall sales data
decomposition = seasonal_decompose(overall_sales, model='additive', period=12)

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(overall_sales, label='Original Data')
plt.title('Original Time Series')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.title('Trend Component')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.title('Seasonality Component')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.title('Residual Component')
plt.legend()

plt.tight_layout()
plt.show()

