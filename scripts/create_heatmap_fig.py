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

def create_heatmap(df):
    # Correlation matrix
    corr_matrix = df.corr()
    corr_abs = corr_matrix.abs()

    # Mask upper triangle + diagonal
    mask = np.triu(np.ones_like(corr_abs, dtype=bool))

    plt.figure(figsize=(10, 8))

    # Colormap
    cmap = sns.light_palette("blue", as_cmap=True)

    ax = sns.heatmap(
        corr_abs,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0, vmax=1,
        annot_kws={'size': 16},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '|Correlation|'}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(16)

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.suptitle("Absolute Correlation of Channel Spend", fontsize=24)
    plt.title("Correlated channels suggest shared marketing effort - some redundant and some synergistic", fontsize=14)

    plt.tight_layout()
    plt.show()

create_heatmap(df)