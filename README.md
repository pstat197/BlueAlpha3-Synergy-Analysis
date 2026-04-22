# Correlation Analysis

Python module utilizing correlation analysis to explore statistical dependency in marketing channels. 

### Overview
- Goal: identify and quantify channel interaction strength on a toy dataset, and select a subset of pairs for later MMM integration.
- Stack: Python, Jupyter notebook.

### Dataset
- Source: BlueAlpha toy dataset

### Methods
- Preprocessing: filter out zero-spend channels.
- Hypothesis testing: Granger causality.
- Modeling: OLS.
- Bias correcting: False Discovery Rate (FDR). 

### Results (highlights)
- Identified Granger causation among channel pairs:
  - amazon_spend -> meta_spend
  - google_spend -> liveintent_spend	
- Found significant interaction confidence intervals:
  - beehiiv_spend + liveintent_spend 

### How to Run
1) Run `interaction_strength_results.ipynb`.

### Structure
- `data/monthly_mocha.csv`: toy dataset
- `interaction_strength_figs`: contains the plots for the notebook
- `results/interaction_strength_results.ipynb`: notebook containing interaction strength analysis
- `scripts/`: contains the scripts for statistical testing and modeling

### Next Steps
- Integrate significant channel interaction pairs with Meridian MMM.
- Prepare ML workflow for the report generator.
