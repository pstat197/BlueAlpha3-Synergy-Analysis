import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_barchart():
    data = {
        'R-squared': [0.87, 0.92, 0.85],
        'MAPE': [0.12, 0.08, 0.14],
        'wMAPE': [0.10, 0.07, 0.11]
    }
    models = ['geometric', 'residualized', 'rectified']
    df = pd.DataFrame(data, index=models)
    
    metrics = df.columns
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        bars = ax.bar(
            x + i * width - width,
            df.loc[model],
            width,
            label=model,
            color=colors[i]
        )
        
        # Optional: add value labels (matches annot_kws size=16)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=16
            )
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    ax.set_ylabel('Value', fontsize=12)
    
    plt.suptitle("Model Performance Comparison", fontsize=24)
    ax.set_title("Comparing model fit and error metrics across specifications", fontsize=14)
    
    ax.legend(title='Model', fontsize=12, title_fontsize=12)
    
    plt.tight_layout()
    plt.show()

create_barchart()