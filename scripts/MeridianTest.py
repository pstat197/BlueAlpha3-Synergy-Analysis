import pandas as pd
import numpy as np

df = pd.read_csv("data/monthly_mocha.csv")
df = df.drop(columns=["date"])

# Remove impressions + zero columns
df = df[[c for c in df.columns if not c.endswith("_impressions")]]
df = df.loc[:, (df != 0).any(axis=0)]

target = "subscriptions"
spend_cols = [c for c in df.columns if c.endswith("_spend")]


def adstock(x, decay):
    result = np.zeros_like(x)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + decay * result[t-1]
    return result

# Apply adstock to each channel
adstocked = pd.DataFrame()
for col in spend_cols:
    adstocked[col] = adstock(df[col].values, decay=0.5) 

def saturation(x, alpha):
    return x / (x + alpha)

sat_df = pd.DataFrame()
for col in spend_cols:
    sat_df[col] = saturation(adstocked[col], alpha=1000)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(sat_df)

X = pd.DataFrame(X_scaled, columns=spend_cols)
y = df[target]

from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X, y)

coefs = pd.Series(model.coef_, index=spend_cols)
print(coefs.sort_values(ascending=False))

# Unsacled to preserve maginitude for contribution analysis
X_unscaled = sat_df.copy()
contribution = X_unscaled.multiply(model.coef_, axis=1)
channel_contribution = contribution.mean().sort_values(ascending=False)

print("\nChannel Contributions:")
print(channel_contribution)