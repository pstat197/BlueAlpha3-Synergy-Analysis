import pandas as pd
from find_interactions import find_interactions_func
from statsmodels.api import OLS, add_constant

def create_residualized_interactions(df):
    """
    Forms pairwise interactions by residualization.
    Requires the DataFrame to have columns with "_spend" in their names and for there to be no zero spend columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The input DataFrame appended with residualized interactions.
    """
    interactions = find_interactions_func(df)
    residualized = df.copy()

    for _, row in interactions.iterrows():
        target = row["target"]
        driver = row["driver"]

        # Interaction term (what we want to residualize)
        y = residualized[driver] * residualized[target]

        # Main effects (what we want to regress out)
        X = residualized[[driver, target]]
        X = add_constant(X)

        # Fit model
        model = OLS(y, X).fit()

        # Create new column with residuals
        col_name = f"{target}_{driver}_residual"

        # Avoid accidental overwrite (optional but safer)
        if col_name in residualized.columns:
            continue

        residualized[col_name] = model.resid

    return residualized

df = pd.read_csv(
  "https://raw.githubusercontent.com/pstat197/BlueAlpha3-Synergy-Analysis/refs/heads/meridian_modeling/data/monthly_mocha.csv"
)
df = df.loc[:, (df != 0).any()]

residualized_df = create_residualized_interactions(df)

print(residualized_df.head())