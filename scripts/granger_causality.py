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

print("\n✅ NEW BLOCK REACHED: starting summary/FDR/visuals...\n")
# ============================================================
# ADD-ON: summarize Granger results + multiple testing (FDR)
#         + visuals (heatmap + network)
# ============================================================

import numpy as np
import pandas as pd

# ---------- 1) Reshape gc_df into a clean long table ----------
# gc_df index is (target, driver) based on your loop:
#   grangercausalitytests(channel_diff[[x1, x2]]) tests whether x2 -> x1
# so: target = x1, driver = x2
long_df = (
    gc_df.reset_index()
        .rename(columns={"level_0": "target", "level_1": "driver"})
        .melt(id_vars=["target", "driver"], var_name="lag", value_name="p_value")
)

# Convert "lag_1_pvalue" -> 1, etc.
long_df["lag"] = long_df["lag"].str.extract(r"lag_(\d+)_pvalue").astype(int)

# ---------- 2) Add min p-value + best lag summary ----------
summary_df = (
    long_df.sort_values(["target", "driver", "p_value"])
           .groupby(["target", "driver"], as_index=False)
           .first()  # after sorting, first row per pair is min p-value & its lag
           .rename(columns={"p_value": "min_pvalue", "lag": "best_lag"})
)

# ---------- 3) Multiple testing correction (Benjamini-Hochberg / FDR) ----------
def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values (q-values) aligned to original order.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

summary_df["fdr_qvalue"] = fdr_bh(summary_df["min_pvalue"].values)
summary_df["sig_p05"] = summary_df["min_pvalue"] < 0.05
summary_df["sig_fdr_05"] = summary_df["fdr_qvalue"] < 0.05

# Sort for convenience (smallest q-values first)
summary_df_sorted = summary_df.sort_values(["fdr_qvalue", "min_pvalue"])

print("\n=== Granger summary (min p-value + best lag + FDR q-value) ===")
print(summary_df_sorted.head(30))

print("\n=== Significant after FDR (q < 0.05) ===")
print(summary_df_sorted[summary_df_sorted["sig_fdr_05"]].head(100))

# Optional: save outputs for teammates
summary_df_sorted.to_csv("granger_summary_with_fdr.csv", index=False)
print("\nSaved: granger_summary_with_fdr.csv")

# ---------- 4) Simple correlation heatmap (original levels, not differenced) ----------
# (This is "do channels move together at the same time?")
try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr = monthly_mocha_channels.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, center=0)
    plt.title("Correlation Heatmap (Channel Levels)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("\n[Heatmap skipped] seaborn/matplotlib not available or error:", e)

# ---------- 5) Directed network graph of significant relationships ----------
# Edge direction is driver -> target (because x2 Granger-causes x1)
try:
    import networkx as nx
    import matplotlib.pyplot as plt

    # Choose which significance rule you want for the graph:
    # Use FDR (recommended). If that yields nothing, switch to raw p<0.05.
    sig_edges = summary_df_sorted[summary_df_sorted["sig_fdr_05"]].copy()

    if sig_edges.empty:
        print("\n[Network note] No edges significant after FDR; falling back to raw p<0.05.")
        sig_edges = summary_df_sorted[summary_df_sorted["sig_p05"]].copy()

    G = nx.DiGraph()
    for _, row in sig_edges.iterrows():
        driver = row["driver"]
        target = row["target"]
        # store lag/p as edge attributes
        G.add_edge(driver, target, best_lag=int(row["best_lag"]), p=float(row["min_pvalue"]), q=float(row["fdr_qvalue"]))

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=1200)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # label edges with best lag
    edge_labels = {(u, v): f"lag {d['best_lag']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title("Granger Network (driver → target), labeled by best lag")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("\n[Network graph skipped] networkx/matplotlib not available or error:", e)

# ---------- 6) (Optional) Quick table to paste into notes/slides ----------
# This is a compact view of the "top relationships"
top_for_notes = summary_df_sorted.head(15)[["driver", "target", "best_lag", "min_pvalue", "fdr_qvalue", "sig_fdr_05"]]
print("\n=== Top relationships (for meeting notes) ===")
print(top_for_notes.to_string(index=False))