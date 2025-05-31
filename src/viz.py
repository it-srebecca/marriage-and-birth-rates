import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def highlight_improvements(df):
    return df.style \
        .background_gradient(
            subset=['Marriage MSE Reduction (%)'],
            cmap='RdYlGn',  # Red = worse, Green = better
            vmin=-100, vmax=100
        ) \
        .background_gradient(
            subset=['Fertility MSE Reduction (%)'],
            cmap='RdYlGn',
            vmin=-100, vmax=100
        ) \
        .format({
            'Marriage MSE (Univariate)': '{:.2f}',
            'Marriage MSE (With Fertility)': '{:.2f}',
            'Marriage MSE Reduction (%)': '{:.1f}%',
            'Fertility MSE (Univariate)': '{:.2f}',
            'Fertility MSE (With Marriage)': '{:.2f}',
            'Fertility MSE Reduction (%)': '{:.1f}%'
        })