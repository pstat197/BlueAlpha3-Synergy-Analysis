import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(
  "https://raw.githubusercontent.com/pstat197/BlueAlpha3-Synergy-Analysis/refs/heads/meridian_modeling/data/monthly_mocha.csv"
)

# clean data
df = df.select_dtypes(include='number')
df = df.filter(regex='(_spend)$')
df = df.loc[:, (df != 0).any()]

def create_heatmap(df, threshold=0.3):
    # Correlation matrix
    corr_matrix = df.corr()

    # Emphasize strength over direction
    corr_abs = corr_matrix.abs()

    plt.figure(figsize=(10, 8))

    # Create colormap and set NaNs (masked values) to gray
    cmap = sns.light_palette("blue", as_cmap=True)
    cmap.set_bad(color='lightgray')

    sns.heatmap(
        corr_abs,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        annot_kws={'size': 8},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '|Correlation|'}
    )

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.title(f"Channel Spend Correlation Strength (|ρ| ≥ {threshold})")

    plt.tight_layout()
    plt.show()

create_heatmap(df)