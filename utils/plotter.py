
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_data_count(data: pd.DataFrame or pd.Series,
                    filename: str):
    plt.plot(data)
    plt.savefig(f"./output/{filename}")
