# Correlation Analysis

Python module utilizing correlation analysis to explore statistical dependency in marketing channels. 

### Overview
- Goal: perform simple correlation analysis on a toy dataset to develop an understanding of channel interactions, and create a working list of channel pairs for later, more sophisticated analysis.
- Stack: Python, Jupyter notebook.

### Dataset
- Source: BlueAlpha toy dataset

### Methods
- Preprocessing: filter out zero-spend channels.
- Modeling: OLS and Ridge with SciKit-Learn.

### Results (highlights)
- Identified potential channel pairs:
  - Meta and LiveIntent
  - Google and Beehiiv
  - Snapchat and TikTok
  - LiveIntent and BeeHiiv
- Observed correlations in lagged residuals

### Next Steps
- Apply Granger causality tests and lagged time series analysis to potential pairs
- Formulate a list of channel pairs ranked by strength
- Calculate confidence intervals to communicate uncertainty

### How to Run
1) Run `correlation_analysis.ipynb`.

### Structure
- `data/monthly_mocha.csv`: toy dataset
- `scripts/correlation_analysis.ipynb`: notebook containing correlation analysis

### Next
- Introduce new models; validate with TensorBoard/MLFlow/etc..
