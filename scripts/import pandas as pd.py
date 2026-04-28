import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import permutations
import io
import contextlib

def find_interactions(df):
    """
    Finds pairwise interactions in a Pandas DataFrame with Granger causality. Corrects for multiple testing with Bonferroni and Benjamini-Hochberg (FDR).

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        list: A list of tuples representing the pairwise interactions.
    """
    # Prepare for Granger

    # Keep only spend columns (remove impressions)
    spend_cols = [c for c in df.columns if "_spend" in c]

    # Create dataset for Granger (exclude target)
    df_gc = df[spend_cols].copy()

    # Difference data to data stationary
    df_gc_diff = df_gc.diff().dropna()

    gc_results = {}

    channels = df_gc_diff.columns

    # Run Granger causality tests
    for x1, x2 in permutations(channels, 2):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            test_result = grangercausalitytests(
                df_gc_diff[[x1, x2]],
                maxlag=3
            )

        p_values = {
            lag: test_result[lag][0]['ssr_ftest'][1]
            for lag in range(1, 4)
        }

        gc_results[(x1, x2)] = p_values

    # Convert to wide DataFrame
    gc_df = pd.DataFrame(gc_results).T
    gc_df.columns = [f"lag_{i}_pvalue" for i in range(1, 4)]

    # Create a long-format DataFrame
    long_df = (
        gc_df.reset_index()
        .rename(columns={"level_0": "target", "level_1": "driver"})
        .melt(
            id_vars=["target", "driver"],
            var_name="lag",
            value_name="p_value"
        )
    )

    # Extract numeric lag
    long_df["lag"] = long_df["lag"].str.extract(r"lag_(\d+)_pvalue").astype(int)

    # Multiple testing corrections
    m = len(long_df)

    # Bonferroni
    long_df["bonferroni_p"] = np.minimum(long_df["p_value"] * m, 1.0)
    long_df["sig_bonf_05"] = long_df["bonferroni_p"] < 0.05

    # Benjamini-Hochberg (FDR)
    pvals = np.asarray(long_df["p_value"].values, dtype=float)

    # Sort p-values and keep track of original order
    order = np.argsort(pvals)
    ranked = pvals[order]

    # Compute BH adjusted values
    q = ranked * len(pvals) / (np.arange(1, len(pvals) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    # Reorder back to original positions
    fdr_q = np.empty_like(q)
    fdr_q[order] = q

    # Assign back to DataFrame
    long_df["fdr_q"] = fdr_q
    long_df["sig_fdr_05"] = long_df["fdr_q"] < 0.05

    # Filter for significant results
    significant = long_df[long_df["sig_fdr_05"]]

    best_lags = (
        significant.sort_values(["target", "driver", "p_value"])
        .groupby(["target", "driver"], as_index=False)
        .first()
    )

    return best_lags

df = pd.read_csv(
  "https://raw.githubusercontent.com/pstat197/BlueAlpha3-Synergy-Analysis/refs/heads/meridian_modeling/data/monthly_mocha.csv"
)
df = df.loc[:, (df != 0).any()]
interactions = find_interactions(df)

print(interactions["driver"].unique().tolist())
