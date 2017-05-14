#!/usr/bin/env python3

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

import os
import itertools

DATA_DIR = "./data"
P1_TRAIN_FILEPATH = os.path.join(DATA_DIR, "p1_data_train.csv")
P1_TEST_FILEPATH = os.path.join(DATA_DIR, "p1_data_test.csv")
X_LABELS = ["Temp1", "Temp2", "Temp3", "Temp4"]
Y_LABEL = "target"

def visualize():
    #opening train dataset
    df = pd.read_csv(P1_TRAIN_FILEPATH)

    #getting indexes for positive/negative labels
    pos_idx = df[df[Y_LABEL] == 1].index.tolist()
    neg_idx = df[df[Y_LABEL] == 0].index.tolist()

    #visualizing each temperature separately
    fig = plt.figure()
    for i, x in enumerate(X_LABELS):
        ax = fig.add_subplot(221 + i)
        ax.scatter(df.ix[pos_idx][x].values, len(pos_idx)*[1], color="g")
        ax.scatter(df.ix[neg_idx][x].values, len(neg_idx)*[0], color="r")
        ax.set_yticks([0, 1])
        ax.yaxis.grid(False)
        ax.set_xlabel(x)
        ax.set_ylabel("Qualidade")
    plt.tight_layout()
    plt.show()

    #visualizing all combinations of temperatures in 3D
    fig = plt.figure()
    for i, (x1, x2, x3) in enumerate(itertools.combinations(X_LABELS, 3)):
        ax = fig.add_subplot(221 + i, projection="3d")
        ax.scatter(df.ix[pos_idx][x1].values, df.ix[pos_idx][x2].values,
            df.ix[pos_idx][x3].values, c="g")
        ax.scatter(df.ix[neg_idx][x1].values, df.ix[neg_idx][x2].values,
            df.ix[neg_idx][x3].values, c="r")
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel(x3)
    plt.tight_layout()
    plt.show()

def main():
    visualize()

if __name__ == "__main__":
    main()
