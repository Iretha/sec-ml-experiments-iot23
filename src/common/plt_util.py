import matplotlib
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def plt_heatmap(df, columns_count, revert=False):
    print("=========> Printing Heatmap with col. count = " + str(columns_count))
    plt.figure(figsize=[columns_count, columns_count])
    sns.heatmap(df, annot=True, fmt='.0%', cmap=sns.cm.rocket_r if revert else sns.cm.rocket)
    plt.show()
