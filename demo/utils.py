"""
Utility funcions for insightdemo.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


class NullModel(BaseEstimator):
    """
    Implements a null model for binary classifier performance, where
    binary labels are assigned randomly to the data.
    """

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        p = np.random.randint(2, size=X.shape[0])
        p.reshape(p.shape[0], 1)
        return p


class NeverDisplaced(BaseEstimator):
    """
    Implements a null model for binary classifier performance, where
    binary labels are assigned randomly to the data.
    """

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        p = np.zeros((len(X), 1), dtype=bool)
        return p


def mathy_header(data):
    """
    Turns column names into markdown latex syntax for nice column header rendering.
    """
    for col in data.columns[:-3]:
        if "{" in col or "}" in col:
            pass
        elif "_" in col and "^" in col:
            superscript_loc = col.index("^") + 1
            subscript_loc = col.index("_")
            updated_name = "$" + col[0:superscript_loc] + "{" + col[superscript_loc:subscript_loc] + "}" + "_{" + col[subscript_loc + 1:] + "}" + "$"
            data.rename(columns={col: updated_name}, inplace=True)
        elif "_" in col:
            subscript_loc = col.index("_") + 1
            updated_name = "$" + col[0:subscript_loc] + "{" + col[subscript_loc:] + "}" + "$"
            data.rename(columns={col: updated_name}, inplace=True)
        else:
            if "^" in col:
                superscript_loc = col.index("^") + 1
                updated_name = "$" + col[0:superscript_loc] + "{" + col[superscript_loc:] + "}" + "$"
                data.rename(columns={col: updated_name}, inplace=True)
            else:
                updated_name = col



def plot_learning_curve(estimator, title, X, y, acceptable_score=None, ylim=None, cv=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Code from: https://jmetzen.github.io/2015-01-29/ml_advice.html
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    acceptable_score : float
        Acceptable model performance

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """
    
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes, scoring="roc_auc")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    if acceptable_score is not None:
        plt.plot([0, plt.xlim()[1]], [acceptable_score, acceptable_score], color='grey', linestyle='--', linewidth=1)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    #plt.savefig(title + "_learning_curve.png")

def plot_roc_curve(fpr_list, tpr_list, name="test", labels=None):
    plt.figure()
    for i, fpr in enumerate(fpr_list):
        plt.plot(fpr, tpr_list[i], linewidth=2, label=labels[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="best")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(name + "_roc.png")



# from: http://collectivesolver.com/5780/how-to-format-bytes-to-kilobytes-megabytes-gigabytes-and-terabytes-in-python
def format_bytes(bytes_num):
    sizes = ["B", "KB", "MB", "GB", "TB"]

    i = 0
    dblbyte = bytes_num

    while (i < len(sizes) and bytes_num >= 1024):
        dblbyte = bytes_num / 1024.0
        i = i + 1
        bytes_num = bytes_num / 1024

    return str(round(dblbyte, 2)) + " " + sizes[i]
