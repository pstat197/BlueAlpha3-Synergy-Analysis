# Import dependencies
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import permutations

# Import and clean data
monthly_mocha_channels = pd.read_csv("data/monthly_mocha.csv").drop(columns=["date", "subscriptions"])

# drop zero-val cols
zero_value_columns = [col for col in monthly_mocha_channels.columns if (monthly_mocha_channels[col] == 0).all()]
monthly_mocha_channels = monthly_mocha_channels.drop(columns=zero_value_columns)

# drop spend cols
impression_cols = [col for col in monthly_mocha_channels.columns if col.endswith("_impressions")]
monthly_mocha_channels = monthly_mocha_channels.drop(columns=impression_cols)

# Ensure stationarity by differencing and dropping NA
channel_diff = monthly_mocha_channels.diff().dropna()

# Store results
gc_results = {}

# Loop through all ordered pairs of channels
channels = channel_diff.columns

for x1, x2 in permutations(channels, 2):
    
    test_result = grangercausalitytests(
        channel_diff[[x1, x2]],
        maxlag=3,
        verbose=False
    )
    
    # Store p-values for each lag (using ssr_ftest)
    p_values = {
        lag: test_result[lag][0]['ssr_ftest'][1]
        for lag in range(1, 4)
    }
    
    gc_results[(x1, x2)] = p_values

# Convert results to DataFrame for easy viewing
gc_df = pd.DataFrame(gc_results).T.rename(columns=lambda x: f"lag_{x}_pvalue")

print(gc_df)