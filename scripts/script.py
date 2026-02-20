# Import depedencies

import pandas as pd


# Import data

df = pd.read_csv("data/monthly_mocha.csv")

print(df.describe())
