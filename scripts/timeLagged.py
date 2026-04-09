import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# Load Data

df = pd.read_csv("data/monthly_mocha.csv")

# Convert and sort date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

#print("Data shape:", df.shape)
#print("Columns:", list(df.columns))

target = "subscriptions"

# Drop impression columns so this matches the other scripts
df = df[[c for c in df.columns if not c.endswith("_impressions")]]

# Keep only spend columns
spend_cols = [col for col in df.columns if col.endswith("_spend")]

# Remove all-zero spend columns
spend_cols = [col for col in spend_cols if (df[col] != 0).any()]

print("\nUsing spend columns:")
print(spend_cols)

df_std = df.copy()

cols_to_scale = spend_cols + [target]

df_std[cols_to_scale] = (
    df_std[cols_to_scale] - df_std[cols_to_scale].mean()
) / df_std[cols_to_scale].std()

# Lagged Correlation Function
def calculate_lagged_correlation(df, target_col, feature_cols, max_lag=6):
    
    correlations = []
    
    for feature in feature_cols:
        for lag in range(0, max_lag + 1):
            
            if lag == 0:
                lagged_feature = df[feature]
            else:
                lagged_feature = df[feature].shift(lag)
            
            valid_idx = ~(df[target_col].isna() | lagged_feature.isna())
            
            if valid_idx.sum() > 2:
                corr, pval = pearsonr(
                    df.loc[valid_idx, target_col],
                    lagged_feature.loc[valid_idx]
                )
                
                correlations.append({
                    'channel': feature,
                    'lag': lag,
                    'correlation': corr,
                    'p_value': pval
                })
    
    return pd.DataFrame(correlations)

# Lag Analysis

lag_results = calculate_lagged_correlation(
    df=df_std,
    target_col=target,
    feature_cols=spend_cols,
    max_lag=6
)

# Bonferroni correction across the 7 tested lags (0 through 6)
n_lags_tested = 7
lag_results["p_value_adj"] = (lag_results["p_value"] * n_lags_tested).clip(upper=1.0)
lag_results["sig_raw_05"] = lag_results["p_value"] < 0.05
lag_results["sig_bonf_05"] = lag_results["p_value_adj"] < 0.05

#print("\nLagged Correlation Results:")
#print(lag_results.head())

# best lag per channel after Bonferroni adjustment

best_lags = (
    lag_results
    .loc[lag_results.groupby('channel')['correlation'].apply(lambda x: x.abs().idxmax())]
    .reset_index(drop=True)
)

print("\nBest Lag Per Channel (with Bonferroni-adjusted p-values):")
print(best_lags[["channel", "lag", "correlation", "p_value", "p_value_adj", "sig_bonf_05"]])

os.makedirs("interaction_strength_figs", exist_ok=True)
best_lags.to_csv("interaction_strength_figs/best_lag_summary.csv", index=False)
lag_results.to_csv("interaction_strength_figs/lagged_correlation_results.csv", index=False)

print("Saved: interaction_strength_figs/best_lag_summary.csv")
print("Saved: interaction_strength_figs/lagged_correlation_results.csv")

# Heat Map

pivot_table = lag_results.pivot(
    index='channel',
    columns='lag',
    values='correlation'
)

print("\nPivot table preview:")
print(pivot_table)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", center=0)

plt.title("Lagged Correlation Heatmap")
plt.xlabel("Lag")
plt.ylabel("Channel")

plt.tight_layout()

plt.savefig("interaction_strength_figs/lagged_heatmap.png", dpi=300, bbox_inches="tight")
print("Heatmap saved as interaction_strength_figs/lagged_heatmap.png")

plt.show()