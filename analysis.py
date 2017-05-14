#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

import os
import itertools
import pickle
import json

#dataset label names
X_LABELS = ["Temp1", "Temp2", "Temp3", "Temp4"]
Y_LABEL = "target"

#directory to store data
DATA_DIR = "./data"
#directory to store models
MODELS_DIR = "./models"
#data/models filepaths
P1_TRAIN_FILEPATH = os.path.join(DATA_DIR, "p1_data_train.csv")
P1_TEST_FILEPATH = os.path.join(DATA_DIR, "p1_data_test.csv")
ESTIMATOR_FILEPATH = os.path.join(MODELS_DIR, "log_reg_estimator.pkl")
X_STATS_FILEPATH = os.path.join(MODELS_DIR, "X_stats.json")
PRED_FILEPATH = os.path.join(DATA_DIR, "p1_predictions.csv")

#parameters for logreg grid search
GRID_SEARCH_PARAMS = {
    "C": [2**n for n in range(-10, 11)],
    "penalty": ["l1", "l2"]
}
#number of folds for cross-validation
CV_N_FOLDS = 3

def pre_process(X, X_stats=None):
    """Pre-processing of input data."""
    stats = {} if X_stats is None else X_stats
    rows, cols = X.shape

    for c, label in zip(range(cols), X_LABELS):
        if not label in stats:
            u = X[c].mean()
            sd = X[c].std()
            stats[label] = {"mean": u, "std": sd}
        X[c] = (X[c] - stats[label]["mean"])/stats[label]["std"]

    return X, stats

def visualize():
    """Visualizes dataset."""
    #opening train dataset
    df = pd.read_csv(P1_TRAIN_FILEPATH)

    #getting indexes for positive/negative labels
    pos_idx = df[df[Y_LABEL] == 1].index.tolist()
    neg_idx = df[df[Y_LABEL] == 0].index.tolist()

    #visualizing each temperature separately
    print("\nvisualizing each temperature...")
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
    print("visualizing temperatures in groups of 3...")
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

def train():
    """Logistic Regression fitting with Grid Search."""

    #opening train dataset
    print("\nloading data from '{}'...".format(P1_TRAIN_FILEPATH), end=" ",
        flush=True)
    df = pd.read_csv(P1_TRAIN_FILEPATH)
    print("done.")

    X = df[X_LABELS].as_matrix()
    y = df[Y_LABEL].as_matrix()

    #pre-processing X data
    print("pre-processing data...", end=" ", flush=True)
    X, X_stats = pre_process(X)
    print("done.")

    #logistic regression estimator
    est = LogisticRegression()

    #performing grid search
    gs = GridSearchCV(est, GRID_SEARCH_PARAMS, cv=CV_N_FOLDS)
    print("performing grid search...", end=" ", flush=True)
    gs.fit(X, y)
    print("done.")

    print("best parameters:", gs.best_params_)
    print("best accuracy: {:.2f}%\n".format(100*gs.best_score_))

    #saving best estimator and X stats
    est = gs.best_estimator_
    print("saving model to '{}'...".format(ESTIMATOR_FILEPATH), end=" ",
        flush=True)
    with open(ESTIMATOR_FILEPATH, "wb") as f:
        pickle.dump(est, f)
    print("done.")

    print("saving X stats to '{}'...".format(X_STATS_FILEPATH), end=" ",
        flush=True)
    with open(X_STATS_FILEPATH, "w") as f:
        json.dump(X_stats, f)
    print("done.")

def test():
    """Estimator testing."""

    #loading data for prediction
    print("\nloading data from '{}'...".format(P1_TEST_FILEPATH), end=" ",
        flush=True)
    df = pd.read_csv(P1_TEST_FILEPATH)
    print("done.")

    X = df[X_LABELS].as_matrix()

    print("loading best estimator from '{}'...".format(ESTIMATOR_FILEPATH),
        end=" ", flush=True)
    with open(ESTIMATOR_FILEPATH, "rb") as f:
        est = pickle.load(f)
    print("done.")

    print("loading X stats from '{}'...".format(X_STATS_FILEPATH),
        end=" ", flush=True)
    with open(X_STATS_FILEPATH, "rb") as f:
        X_stats = json.load(f)
    print("done.")

    #pre-processing X data
    print("pre-processing data...", end=" ", flush=True)
    X, __ = pre_process(X, X_stats)
    print("done.")

    #estimating class from test set
    print("\npredicting...", end=" ", flush=True)
    y = est.predict(X)
    print("done. some results:")
    df["target"] = y
    print(df[:10])

    #saving results
    print("\nsaving results to '{}'...".format(PRED_FILEPATH), end=" ",
        flush=True)
    pred_df = pd.DataFrame({"target": y})
    pred_df.to_csv(PRED_FILEPATH, index=False)
    print("done.\n")

def main():
    visualize()
    train()
    test()

if __name__ == "__main__":
    main()
