"""
Scikit-learn Isolation Forest format:
Labels:
    Normal: 1
    Anomaly: -1

Scoring:
    Normal: negative score close to 0
    Anomaly: negative score close to -1

--------------------------------------------

In some parts of the code, it was converted to:
Labels:
    Normal: 0
    Anomaly: 1

Scoring:
    Normal: positive score close to 0
    Anomaly: positive score close to 1
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from data.dataset_utils import Subjects
from thresholds import test_supervised_acc_threshold, train_supervised_acc_threshold, fixed_percentile_threshold, mixmod_threshold, filter_threshold, karch_threshold, eb_threshold


def run_train_test_split(features, params, subject_specific=False):
    """
    Train and test the Isolation Forest model with the given parameters on the given features.

    Returns
    -------
    Dict
        Accuracy, precision, recall and AUC score.
    """
    if not subject_specific:
        train_subjects = Subjects.train
        test_subjects = Subjects.val
        X_train = features[features['subject_id'].isin(train_subjects)].iloc[:, :-2]
        y_train = features[features['subject_id'].isin(train_subjects)]['class'].replace({0: 1, 4: -1})
        X_test = features[features['subject_id'].isin(test_subjects)].iloc[:, :-2]
        y_test = features[features['subject_id'].isin(test_subjects)]['class'].replace({0: 1, 4: -1})

        X_train_normal = X_train[y_train == 1]

        clf = IsolationForest(random_state=42, n_jobs=-1, **params)
        clf.fit(X_train_normal)
        y_pred = clf.predict(X_test)
        y_test_scores = clf.score_samples(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_test_scores))
    else:
        np.random.seed(4)
        subject_ids = Subjects.val
        accs = []
        precs = []
        recs = []
        aucs = []
        for subject_id in subject_ids:
            subject_data = features[features['subject_id'] == subject_id]
            normals = subject_data[subject_data['class'] == 0].index.values
            anomalies = subject_data[subject_data['class'] == 4].index.values
            
            # 15 normals for training, 5 normals and 5 anomalies for testing
            train_normals = np.random.choice(normals, size=15, replace=False)
            test_normals = np.setdiff1d(normals, train_normals)
            test_anomalies = np.random.choice(anomalies, size=5, replace=False)
            
            train_indices = train_normals
            test_indices = np.concatenate([test_normals, test_anomalies])
            X_train = features.iloc[train_indices, :-2]
            y_train = features.iloc[train_indices]['class'].replace({0: 1, 4: -1})
            X_test = features.iloc[test_indices, :-2]
            y_test = features.iloc[test_indices]['class'].replace({0: 1, 4: -1})

            X_train_normal = X_train[y_train == 1]

            clf = IsolationForest(random_state=42, n_jobs=-1, **params)
            clf.fit(X_train_normal)
            y_pred = clf.predict(X_test)
            y_test_scores = clf.score_samples(X_test)

            accs.append(float(accuracy_score(y_test, y_pred)))
            precs.append(float(precision_score(y_test, y_pred, zero_division=0)))
            recs.append(float(recall_score(y_test, y_pred)))
            aucs.append(float(roc_auc_score(y_test, y_test_scores)))

        acc = np.mean(accs)
        prec = np.mean(precs)
        rec = np.mean(recs)
        auc = np.mean(aucs)

    return {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'auc': auc
    }



def plot_feature_importance(features, aucs, figsize=(10, 10)):
    """
    Plot the feature importance of the features based on the list of AUC scores per feature.
    """
    sorted_indices = np.argsort(aucs)[::1]
    
    plt.figure(figsize=figsize)
    plt.barh(features.columns[:-2][sorted_indices], np.array(aucs)[sorted_indices])
    plt.ylabel("Feature")
    plt.xlabel("AUC")
    plt.show()



def grid_search(features, param_grid, subject_specific=False):
    """
    Perform a grid search to find the best parameters for the Isolation Forest model.
    """
    if not subject_specific:
        train_subjects = Subjects.train
        test_subjects = Subjects.val
        X_train = features[features['subject_id'].isin(train_subjects)].iloc[:, :-2]
        y_train = features[features['subject_id'].isin(train_subjects)]['class'].replace({0: 1, 4: -1})
        X_test = features[features['subject_id'].isin(test_subjects)].iloc[:, :-2]
        y_test = features[features['subject_id'].isin(test_subjects)]['class'].replace({0: 1, 4: -1})

        X_train_normal = X_train[y_train == 1]

        X = np.vstack((X_train_normal, X_test))
        y = np.hstack((np.ones(len(X_train_normal)), y_test))
        test_fold = [-1] * len(X_train_normal) + [0] * len(X_test)
        cv = PredefinedSplit(test_fold=test_fold)
    else:
        np.random.seed(4)
        subject_ids = Subjects.val
        cv = []
        
        for subject_id in subject_ids:
            subject_data = features[features['subject_id'] == subject_id]
            normals = subject_data[subject_data['class'] == 0].index.values
            anomalies = subject_data[subject_data['class'] == 4].index.values
            
            # 15 normals for training, 5 normals and 5 anomalies for testing
            train_normals = np.random.choice(normals, size=15, replace=False)
            test_normals = np.setdiff1d(normals, train_normals)
            test_anomalies = np.random.choice(anomalies, size=5, replace=False)
            
            train_indices = train_normals
            test_indices = np.concatenate([test_normals, test_anomalies])
            cv.append((train_indices, test_indices))
        
        X = features.drop(columns=['subject_id', 'class'])
        y = features['class'].replace({0: 1, 4: -1})

    grid_search = GridSearchCV(IsolationForest(random_state=4, n_jobs=-1), param_grid, scoring='roc_auc', n_jobs=-1, cv=cv)
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best AUC: {grid_search.best_score_}")




def run_loso_CV(df, params, percentile, direct_thresholding=False):
    """
    Leave one subject out cross validation for the isolation forest model.

    Parameters
    ----------
    df : pd.DataFrame
        The features dataframe.
    params : dict
        The parameters for the isolation forest model.
    percentile : float
        The percentile for the percentile threshold function.
    direct_thresholding : bool
        If True, the anomaly score is the first feature of the dataframe.

    Returns
    -------
    auc_scores : list
        List of AUC scores for each subject.
    threshold_scores : list
        List of dictionaries containing threshold metrics (accuracy, precision, recall) for each threshold function.
    scores_normal : list
        List of normal sample scores for each subject.
    scores_anomaly : list
        List of anomaly sample scores for each subject.
        
    """
    subjects = df['subject_id'].unique()
    scores_normal = []
    scores_anomaly = []
    auc_scores = []
    threshold_scores = [{'func': 'default_threshold', 'threshold': [], 'acc': [], 'prec': [], 'rec': []}, {'func': test_supervised_acc_threshold, 'threshold': [], 'acc': [], 'prec': [], 'rec': [], 'percentile':[]}, {'func': train_supervised_acc_threshold, 'threshold': [], 'acc': [], 'prec': [], 'rec': [], 'percentile':[]}]

    for t in [fixed_percentile_threshold, mixmod_threshold, filter_threshold, karch_threshold, eb_threshold]:
        threshold_scores.append({'func': t , 'threshold': [], 'acc': [], 'prec': [], 'rec': []})

    for subject in subjects:
        X_train = df[df['subject_id'] != subject].iloc[:, :-2]
        y_train = df[df['subject_id'] != subject]['class'].replace({0: 1, 4: -1})
        X_test = df[df['subject_id'] == subject].iloc[:, :-2]
        y_test = df[df['subject_id'] == subject]['class'].replace({0: 1, 4: -1})

        X_train_normal = X_train[y_train == 1]
        X_train = pd.concat([X_train_normal, X_train[y_train == -1]])
        y_train = pd.concat([y_train[y_train == 1], y_train[y_train == -1]])
        X_test_normal = X_test[y_test == 1]
        X_test_anomaly = X_test[y_test == -1]

        if not direct_thresholding:
            clf = IsolationForest(random_state=42, n_jobs=-1, **params)
            clf.fit(X_train_normal)

            # predictions default threshold
            y_test_pred = clf.predict(X_test)

            y_train_normal_scores = clf.score_samples(X_train_normal)
            y_train_scores = np.concatenate([y_train_normal_scores, clf.score_samples(X_train[y_train == -1])])
            y_test_normal_scores = clf.score_samples(X_test_normal)
            y_test_anomaly_scores = clf.score_samples(X_test_anomaly)
            y_test_scores = clf.score_samples(X_test)

            # auc score
            auc_scores.append(roc_auc_score(y_test, y_test_scores))

            # default threshold function
            threshold_scores[0]['threshold'].append(np.nan if direct_thresholding else -clf.offset_)
            threshold_scores[0]['acc'].append(accuracy_score(y_test, y_test_pred))
            threshold_scores[0]['prec'].append(precision_score(y_test, y_test_pred, zero_division=0))
            threshold_scores[0]['rec'].append(recall_score(y_test, y_test_pred))

        else:
            # first feature is directly the anomaly score
            y_test_pred = np.zeros(len(y_test))
            y_train_normal_scores = X_train_normal.iloc[:, 0].values
            y_train_scores = np.concatenate([y_train_normal_scores, X_train[y_train == -1].iloc[:, 0].values])
            y_test_normal_scores = X_test_normal.iloc[:, 0].values
            y_test_anomaly_scores = X_test_anomaly.iloc[:, 0].values
            y_test_scores = X_test.iloc[:, 0].values

            # normalize
            min_score = np.min(y_train_normal_scores)
            max_score = np.max(y_train_normal_scores)
            y_train_normal_scores = -(y_train_normal_scores - min_score) / (max_score - min_score)
            y_train_scores = -(y_train_scores - min_score) / (max_score - min_score)
            y_test_normal_scores = -(y_test_normal_scores - min_score) / (max_score - min_score)
            y_test_anomaly_scores = -(y_test_anomaly_scores - min_score) / (max_score - min_score)
            y_test_scores = -(y_test_scores - min_score) / (max_score - min_score)
            
            # auc score
            auc_scores.append(roc_auc_score(y_test, y_test_scores))

        # reformat
        y_train.replace({1: 0, -1: 1}, inplace=True)
        y_test.replace({1: 0, -1: 1}, inplace=True)

        # threshold functions
        for t in threshold_scores:
            if t['func'] == 'default_threshold':
                continue
            elif t['func'] == test_supervised_acc_threshold:
                threshold = t['func'](-y_test_scores, y_test)
                t['percentile'].append(np.sum(-y_test_normal_scores < threshold) / len(y_test_normal_scores) * 100)
                scores_normal.append(-y_test_normal_scores - threshold)
                scores_anomaly.append(-y_test_anomaly_scores - threshold)

            elif t['func'] == train_supervised_acc_threshold:
                threshold = t['func'](-y_train_scores, y_train)
                t['percentile'].append(np.sum(-y_train_normal_scores < threshold) / len(y_train_normal_scores) * 100)
                
            elif t['func'] == fixed_percentile_threshold:
                threshold = t['func'](-y_train_normal_scores, percentile)
            else:
                threshold = t['func'](-y_train_normal_scores)
                
            y_test_pred = -y_test_scores > threshold
            t['threshold'].append(threshold)
            t['acc'].append(accuracy_score(y_test, y_test_pred))
            t['prec'].append(precision_score(y_test, y_test_pred, zero_division=0))
            t['rec'].append(recall_score(y_test, y_test_pred))

    return auc_scores, threshold_scores, scores_normal, scores_anomaly




def run_subject_specific_cv(df, params, percentile):
    """
    Subject specific 4-fold cross validation for the isolation forest model.

    Parameters
    ----------
    df : pd.DataFrame
        The features dataframe.
    params : dict
        The parameters for the isolation forest model.
    percentile : float
        The percentile for the percentile threshold function.

    Returns
    -------
    auc_scores : list
        List of AUC scores for each fold.
    threshold_scores : list
        List of dictionaries containing threshold metrics (accuracy, precision, recall) for each threshold function.
    scores_normal : list
        List of normal sample scores for each fold.
    scores_anomaly : list
        List of anomaly sample scores for each fold.
    """
    subjects = df['subject_id'].unique()
    auc_scores = []
    scores_normal = []
    scores_anomaly = []
    threshold_scores = [
        {'func': 'default_threshold', 'threshold': [], 'acc': [], 'prec': [], 'rec': []}, {'func': test_supervised_acc_threshold, 'threshold': [], 'acc': [], 'prec': [], 'rec': [], 'percentile':[]}, {'func': train_supervised_acc_threshold, 'threshold': [], 'acc': [], 'prec': [], 'rec': [], 'percentile': []}
    ]
    for t in [fixed_percentile_threshold, mixmod_threshold, filter_threshold, karch_threshold, eb_threshold]:
        threshold_scores.append({'func': t, 'threshold': [], 'acc': [], 'prec': [], 'rec': []})

    for subject in subjects:
        subject_data = df[df['subject_id'] == subject]
        normals = subject_data[subject_data['class'] == 0]
        anomalies = subject_data[subject_data['class'] == 4]

        normal_indices = normals.index.values
        anomaly_indices = anomalies.index.values

        np.random.seed(42)
        np.random.shuffle(normal_indices)
        np.random.shuffle(anomaly_indices)

        # Split normals and anomalies into 4 folds
        normal_folds = np.array_split(normal_indices, 4)
        anomaly_folds = np.array_split(anomaly_indices, 4)

        for fold_idx in range(4):
            test_normals_idx = normal_folds[fold_idx]
            train_normals_idx = np.setdiff1d(normal_indices, test_normals_idx)
            test_anomalies_idx = anomaly_folds[fold_idx]
            train_anomalies_idx = np.setdiff1d(anomaly_indices, test_anomalies_idx)
            test_idx = np.concatenate([test_normals_idx, test_anomalies_idx])

            X_train_normal = df.loc[train_normals_idx].drop(columns=['subject_id', 'class'])
            X_train_anomaly = df.loc[train_anomalies_idx].drop(columns=['subject_id', 'class'])
            X_train = pd.concat([X_train_normal, X_train_anomaly])
            y_train = pd.Series(
                [1] * len(train_normals_idx) + [-1] * len(train_anomalies_idx),
                index=list(train_normals_idx) + list(train_anomalies_idx)
            )
            X_test = df.loc[test_idx].drop(columns=['subject_id', 'class'])
            y_test = df.loc[test_idx]['class'].replace({0: 1, 4: -1})
            X_test_normal = df.loc[test_normals_idx].drop(columns=['subject_id', 'class'])
            X_test_anomaly = df.loc[test_anomalies_idx].drop(columns=['subject_id', 'class'])

            # Fit the Isolation Forest model
            clf = IsolationForest(random_state=42, n_jobs=-1, **params)
            clf.fit(X_train_normal)

            # Predictions using default threshold
            y_test_pred = clf.predict(X_test)

            # Score samples
            y_train_normal_scores = clf.score_samples(X_train_normal)
            y_train_scores = np.concatenate([
                y_train_normal_scores,
                clf.score_samples(X_train[y_train == -1])
            ])
            y_test_normal_scores = clf.score_samples(X_test_normal)
            y_test_anomaly_scores = clf.score_samples(X_test_anomaly)
            y_test_scores = clf.score_samples(X_test)

            # AUC score
            auc_scores.append(roc_auc_score(y_test, y_test_scores))

            # Default threshold function
            threshold_scores[0]['threshold'].append(-clf.offset_)
            threshold_scores[0]['acc'].append(accuracy_score(y_test, y_test_pred))
            threshold_scores[0]['prec'].append(precision_score(y_test, y_test_pred, zero_division=0))
            threshold_scores[0]['rec'].append(recall_score(y_test, y_test_pred))

            # Reformat labels for threshold functions
            y_train.replace({1: 0, -1: 1}, inplace=True)
            y_test.replace({1: 0, -1: 1}, inplace=True)

            # Threshold functions
            for t in threshold_scores:
                if t['func'] == 'default_threshold':
                    continue
                elif t['func'] == test_supervised_acc_threshold:
                    threshold = t['func'](-y_test_scores, y_test)
                    t['percentile'].append(
                        np.sum(-y_test_normal_scores < threshold) / len(y_test_normal_scores) * 100
                    )
                    scores_normal.append(-y_test_normal_scores - threshold)
                    scores_anomaly.append(-y_test_anomaly_scores - threshold)
                    
                elif t['func'] == train_supervised_acc_threshold:
                    threshold = t['func'](-y_train_scores, y_train)
                    t['percentile'].append(
                        np.sum(-y_train_normal_scores < threshold) / len(y_train_normal_scores) * 100
                    )

                elif t['func'] == fixed_percentile_threshold:
                    threshold = t['func'](-y_train_normal_scores, percentile)
                else:
                    threshold = t['func'](-y_train_normal_scores)

                y_test_pred = -y_test_scores > threshold
                t['threshold'].append(threshold)
                t['acc'].append(accuracy_score(y_test, y_test_pred))
                t['prec'].append(precision_score(y_test, y_test_pred, zero_division=0))
                t['rec'].append(recall_score(y_test, y_test_pred))

    return auc_scores, threshold_scores, scores_normal, scores_anomaly




def plot_ranked_feature_selection(features, feature_importance, params, signal='all', subject_specific=False):
    """
    Iteratively add features based on the given importance/ranking list. Plot the AUC, Accuracy, Precision and Recall for each iteration.
    """
    stats = []
    features_subset = []
    for i in range(1, len(feature_importance)+1):
        # if signal starts with the given signal or 'all'
        if signal == 'all' or feature_importance[i-1].startswith(signal):
            features_subset.append(feature_importance[i-1])
        else:
            continue
        results = run_train_test_split(features[features_subset + ['subject_id', 'class']], params, subject_specific)
        stats.append((feature_importance[i-1], results))
    # Extracting metrics and feature names
    features = [stat[0] for stat in stats]
    acc_list = [stat[1]['acc'] for stat in stats]
    prec_list = [stat[1]['prec'] for stat in stats]
    rec_list = [stat[1]['rec'] for stat in stats]
    auc_list = [stat[1]['auc'] for stat in stats]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(range(len(features)), auc_list, label='AUC', color='tab:red', marker='o')
    ax.plot(range(len(features)), acc_list, label='Accuracy', color='tab:blue', marker='s')
    ax.plot(range(len(features)), rec_list, label='Recall', color='tab:orange', marker='x')
    ax.plot(range(len(features)), prec_list, label='Precision', color='tab:green', marker='^')

    ax.set_ylabel('Score')
    ax.set_xlabel('Features')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontdict={'family': 'monospace'})

    # Vertical line for max AUC
    ax.axvline(x=np.argmax(auc_list), color='tab:red', linestyle='--', label='Number of Features for max AUC', alpha=0.9)

    # Second x-axis for the count of features
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(features)))
    ax2.set_xticklabels([str(i + 1) if i % 2 != 0 else '' for i in range(len(features))])
    ax2.set_xlabel('Number of Features')

    ax.legend(loc='upper right')
    plt.ylim(top=0.95)
    plt.show()

    print(f'Max AUC: {max(auc_list)}')
    print(f'Max Accuracy: {max(acc_list)}')




def plot_pca_feature_selection(features, params, components):
    """
    Plot the AUC, Accuracy, Precision and Recall for different number of PCA components.
    """
    stats = []
    for component in components:
        pca = PCA(n_components=component)
        pca_data = pca.fit_transform(features.iloc[:, :-2])
        pca_data = pd.DataFrame(pca_data)
        pca_data['class'] = features['class']
        pca_data['subject_id'] = features['subject_id']
        results = run_train_test_split(pca_data, params)
        stats.append((component, results))
    # Extracting metrics and feature names
    components = [stat[0] for stat in stats]
    acc_list = [stat[1]['acc'] for stat in stats]
    prec_list = [stat[1]['prec'] for stat in stats]
    rec_list = [stat[1]['rec'] for stat in stats]
    auc_list = [stat[1]['auc'] for stat in stats]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(components, auc_list, label='AUC', color='tab:red', marker='o')
    ax.plot(components, acc_list, label='Accuracy', color='tab:blue', marker='s')
    ax.plot(components, rec_list, label='Recall', color='tab:orange', marker='x')
    ax.plot(components, prec_list, label='Precision', color='tab:green', marker='^')

    # Vertical lines for max AUC and Accuracy
    ax.axvline(x=(np.argmax(auc_list)+1), color='tab:red', linestyle='--', label='Number of PCA Components for max AUC', alpha=0.9)
    ax.axvline(x=(np.argmax(acc_list)+1), color='tab:blue', linestyle='--', label='Number of PCA Components for max Accuracy', alpha=0.9)

    ax.set_ylabel('Score')
    ax.set_xlabel('Number of PCA Components')
    ax.set_xticks(components)
    ax.legend(loc='upper right')
    plt.ylim(top=0.90)
    plt.show()

    print(f'Max AUC: {max(auc_list)}')
    print(f'Max Accuracy: {max(acc_list)}')