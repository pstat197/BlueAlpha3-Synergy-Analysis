import numpy as np
import pandas as pd
from find_interactions import find_interactions_func

def create_geometric_mean_interactions(df):
    """
    Forms pairwise interactions using geometric means.
    Requires the DataFrame to have columns with "_spend" in their names and no zero-spend columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with geometric interaction terms appended.
    """
    interactions = find_interactions_func(df)
    geometric_mean_interactions = df.copy()

    for _, row in interactions.iterrows():
        target = row["target"]
        driver = row["driver"]

        # Safety: ensure columns exist
        if target not in df.columns or driver not in df.columns:
            continue

        # Compute geometric mean for spend interactions
        spend_interaction_col = f"{target}_x_{driver}_geom"
        geometric_mean_interactions[spend_interaction_col] = np.sqrt(
            geometric_mean_interactions[target] * geometric_mean_interactions[driver]
        )

        target = row["target"].replace("_spend", "_impressions")
        driver = row["driver"].replace("_spend", "_impressions")

        # Safety: ensure columns exist
        if target not in df.columns or driver not in df.columns:
            continue

        # Compute geometric mean for impression interactions
        impression_interaction_col = f"{target}_x_{driver}_geom"
        geometric_mean_interactions[impression_interaction_col] = np.sqrt(
            geometric_mean_interactions[target] * geometric_mean_interactions[driver]
        )

    return geometric_mean_interactions

df = pd.read_csv(
  "https://raw.githubusercontent.com/pstat197/BlueAlpha3-Synergy-Analysis/refs/heads/meridian_modeling/data/monthly_mocha.csv"
)
df = df.loc[:, (df != 0).any()]

geometric_mean_df = create_geometric_mean_interactions(df)

print(geometric_mean_df.head())