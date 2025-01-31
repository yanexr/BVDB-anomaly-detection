import numpy as np
from sklearn.metrics import accuracy_score
from pythresh.thresholds.mixmod import MIXMOD
from pythresh.thresholds.filter import FILTER
from pythresh.thresholds.karch import KARCH
from pythresh.thresholds.eb import EB


def train_supervised_acc_threshold(y_scores, y_true):
    best_theshold = None
    highest_acc = 0

    for threshold in y_scores:
        preds = y_scores > threshold
        acc = accuracy_score(y_true, preds)
        if acc > highest_acc:
            highest_acc = acc
            best_theshold = threshold

    return best_theshold

def test_supervised_acc_threshold(y_scores, y_true):
    return train_supervised_acc_threshold(y_scores, y_true)

def fixed_percentile_threshold(y_scores, percentile=85):
    threshold = np.percentile(y_scores, percentile)
    return threshold

def mixmod_threshold(y_scores):
    mixmod = MIXMOD()
    mixmod.eval(y_scores)
    return mixmod.thresh_

def filter_threshold(y_scores):
    filter = FILTER()
    filter.eval(y_scores)
    return filter.thresh_

def karch_threshold(y_scores):
    karch = KARCH()
    karch.eval(y_scores)
    return karch.thresh_

def eb_threshold(y_scores):
    eb = EB()
    eb.eval(y_scores)
    return eb.thresh_