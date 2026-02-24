import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from itertools import combinations

df = pd.read_csv("data/monthly_mocha.csv")

df = df.drop(columns=["date"])

# Drop impressions
df = df[[c for c in df.columns if not c.endswith("_impressions")]]

# Drop all-zero columns
df = df.loc[:, (df != 0).any(axis=0)]

target = "subscriptions"
spend_cols = [c for c in df.columns if c.endswith("_spend")]

scaler = StandardScaler()
df[spend_cols] = scaler.fit_transform(df[spend_cols])

# correlation matrix
corr_matrix = df[spend_cols].corr().abs()

np.fill_diagonal(corr_matrix.values, 0)

corr_pairs = (
    corr_matrix.unstack()
    .sort_values(ascending=False)
)

# Remove duplicates 
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) <
                        corr_pairs.index.get_level_values(1)]

# Select top 6 pairs
top_pairs = corr_pairs.head(6).index.tolist()

print("Top 6 correlated spend pairs:")
print(top_pairs)

print("Number of weeks:", df.shape[0])

for a, b in top_pairs:
    df[f"{a}_x_{b}"] = df[a] * df[b]

interaction_terms = [f"{a}_x_{b}" for a,b in top_pairs]
features = spend_cols + interaction_terms

X = sm.add_constant(df[features])
y = df["subscriptions"]

model = sm.OLS(y, X).fit()

print(model.summary())

coef_series = model.params[interaction_terms]

ranked = coef_series.abs().sort_values(ascending=False)

print("\nRanked Interaction Effects (absolute size):")
print(ranked)

ci = model.conf_int().loc[interaction_terms]

interaction_summary = pd.DataFrame({
    "coef": model.params.loc[interaction_terms],
    "lower_95": ci[0],
    "upper_95": ci[1]
})

interaction_summary["abs_coef"] = interaction_summary["coef"].abs()
interaction_summary = interaction_summary.sort_values("abs_coef", ascending=False)

print("\nInteraction Effects with 95% Confidence Intervals:")
print(interaction_summary)



# After including the interaction terms, the model achieved an R² of 0.845 (Adjusted R² = 0.808), meaning it explains about 84.5% of the variation in subscriptions. This is a modest improvement over the main-effects-only model, suggesting that adding interactions does capture some additional cross-channel behavior. When ranking the interaction terms by absolute coefficient size, the largest effects appear for Beehiiv × Google (−737), Beehiiv × LiveIntent (+612), and LiveIntent × Meta (−250). However, once we examine the confidence intervals, only the Beehiiv × LiveIntent interaction remains statistically significant at the 95% level, indicating clear positive synergy between those two channels. The Beehiiv × Google and LiveIntent × Meta effects are fairly large in magnitude, but their intervals cross zero, so they should be interpreted as suggestive rather than definitive evidence of cannibalization or diminishing returns. The remaining interaction terms are relatively small and highly uncertain. Overall, this suggests that while several channel pairs show directional interaction signals, Beehiiv and LiveIntent stand out as the only pairing with strong and statistically reliable cross-channel lift in this dataset.