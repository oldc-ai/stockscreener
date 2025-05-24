import pandas as pd

# Read the input CSV file
df = pd.read_csv('ten_x_stocks_with_fundamentals.csv')

# Convert marketCap to numeric, handling any non-numeric values
df['marketCap'] = pd.to_numeric(df['marketCap'], errors='coerce')

# Filter stocks with market cap >= 2 billion (2B = 2,000,000,000)
filtered_df = df[df['marketCap'] >= 2_000_000_000]

# Save the filtered data to a new CSV file
filtered_df.to_csv('ten_x_stocks_with_fundamentals_filtered.csv', index=False)

print(f"Original number of stocks: {len(df)}")
print(f"Number of stocks after filtering: {len(filtered_df)}")
print("Filtered data has been saved to 'ten_x_stocks_with_fundamentals_filtered.csv'")
