import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
import matplotlib.pyplot as plt

from data.dataset_utils import load_data, preprocess, Subjects, prepare_data, get_dataloader, get_dataloaders_subject_specific
from thresholds import test_supervised_acc_threshold, train_supervised_acc_threshold, fixed_percentile_threshold, mixmod_threshold, filter_threshold, karch_threshold, eb_threshold


def leave_one_subject_out_CV(model, config, signals, batch_size, mp_filter, resample_length=2816, emg_envelope=False):
    """
    Perform leave-one-subject-out cross-validation for the given model and configuration.
    """
    evals_test = []
    evals_train = []
    df = load_data(filtered=True)
    df = preprocess(df, resample_length=resample_length, emg_envelope=emg_envelope)

    for subject in Subjects.all:
        # train data of class 0
        train_data_normal, _ = prepare_data(df, signals, classes=[0], leave_one_out_subject=subject, mp_filter=mp_filter)
        train_loader_normal = get_dataloader(train_data_normal, batch_size=batch_size, shuffle=True)

        # test data of class 0 and 4
        _, test_data = prepare_data(df, signals, classes=[0, 4], leave_one_out_subject=subject, mp_filter=mp_filter)
        test_loader = get_dataloader(test_data, batch_size=batch_size, shuffle=False)

        # train data of class 0 and 4 to select a threshold
        _, test_train_data = prepare_data(df, signals, classes=[0, 4], leave_one_in_subject=subject, mp_filter=mp_filter)
        test_train_loader = get_dataloader(test_train_data, batch_size=batch_size, shuffle=False)

        m = model(config)
        m.train_model(train_loader_normal)
        test_eval = m.evaluate_model(test_loader)
        evals_test.append(test_eval)
        test_train_eval = m.evaluate_model(test_train_loader)
        evals_train.append(test_train_eval)

    return evals_test, evals_train



def subject_specific_CV(model, subjects, config, signal, batch_size, mp_filter, resample_length=None, emg_envelope=False):
    """
    Perform subject specific 4-fold cross validation for the given model and configuration.
    """
    evals_test = []
    evals_train = []
    df = load_data(filtered=True)
    df = preprocess(df, resample_length=resample_length, emg_envelope=False)

    for subject in subjects:
        for i in range(4):
            train_loader, train_loader_normal, test_loader = get_dataloaders_subject_specific(df, signal, batch_size=batch_size, subject=subject, mp_filter=mp_filter, fold=i)
            m = model(config)
            m.train_model(train_loader_normal)

            test_eval = m.evaluate_model(test_loader)
            evals_test.append(test_eval)
            test_train_eval = m.evaluate_model(train_loader)
            evals_train.append(test_train_eval)

    return evals_test, evals_train



def report(evals_test, evals_train, percentile):
    """
    Report the results of the evaluation

    Returns a DataFrame with the following columns:
    - method (name of the thresholding method)
    - acc.avg
    - acc.std
    - prec.avg
    - prec.std
    - rec.avg
    - rec.std
    - auc.avg
    - auc.std
    - threshold.avg
    - threshold.std
    - percentile.avg
    - percentile.std
    """
    results = []
    scores_normal = []
    scores_anomaly = []
    for t in [test_supervised_acc_threshold, train_supervised_acc_threshold, fixed_percentile_threshold, mixmod_threshold, filter_threshold, karch_threshold, eb_threshold]:
        results.append({'func': t, 'threshold': [], 'acc': [], 'prec': [], 'rec': [], 'auc': [], 'percentile':[]})

    for eval_test, eval_train in zip(evals_test, evals_train):
        y_test_scores = eval_test['y_scores']
        y_test = eval_test['y_true']
        y_test_scores_normal = np.array(y_test_scores)[np.array(y_test) == 0]
        y_test_scores_anomaly = np.array(y_test_scores)[np.array(y_test) == 1]

        y_train_scores = eval_train['y_scores']
        y_train = eval_train['y_true']
        y_train_scores_normal = np.array(y_train_scores)[np.array(y_train) == 0]

        for t in results:
            if t['func'] == test_supervised_acc_threshold:
                threshold = t['func'](y_test_scores, y_test)
                t['percentile'].append(np.sum(y_test_scores_normal < threshold) / len(y_test_scores_normal) * 100)
            elif t['func'] == train_supervised_acc_threshold:
                threshold = t['func'](y_train_scores, y_train)
                t['percentile'].append(np.sum(y_train_scores_normal < threshold) / len(y_train_scores_normal) * 100)
                scores_normal.append(y_test_scores_normal - threshold)
                scores_anomaly.append(y_test_scores_anomaly - threshold)
            elif t['func'] == fixed_percentile_threshold:
                threshold = t['func'](y_train_scores_normal, percentile)
            else:
                threshold = t['func'](y_train_scores_normal)
            
            y_test_pred = np.array(y_test_scores) > threshold
            t['threshold'].append(threshold)
            t['acc'].append(accuracy_score(y_test, y_test_pred))
            t['prec'].append(precision_score(y_test, y_test_pred))
            t['rec'].append(recall_score(y_test, y_test_pred))
            t['auc'].append(roc_auc_score(y_test, y_test_scores))
                
    result_list = []
    for t in results:
        result_list.append({
            'method': t['func'].__name__,
            'acc.avg': np.mean(t['acc']),
            'acc.std': np.std(t['acc']),
            'prec.avg': np.mean(t['prec']),
            'prec.std': np.std(t['prec']),
            'rec.avg': np.mean(t['rec']),
            'rec.std': np.std(t['rec']),
            'auc.avg': np.mean(t['auc']),
            'auc.std': np.std(t['auc']),
            'threshold.avg': np.mean(t['threshold']),
            'threshold.std': np.std(t['threshold']),
            'percentile.avg': np.mean(t['percentile']),
            'percentile.std': np.std(t['percentile'])
        })

    df = pd.DataFrame(result_list)
    df.columns = pd.MultiIndex.from_tuples([col.split('.') for col in df.columns])
    return df





########################################################################



# Function to sample a configuration
def sample_config(search_space):
    return {key: random.choice(values) for key, values in search_space.items()}

# Function for random search
def random_search(model, search_space, train_loader_normal, test_loader, iterations=2):
    best_config = None
    best_val_auc = -np.inf
    results = []
    
    for i in range(iterations):
        # Sample hyperparameters
        config = sample_config(search_space)
        print(f"Testing configuration {i+1}: {config}")
        
        # Initialize and train the model
        model = model(config)
        _, _, _, val_auc_scores, _ = model.train_model(
            train_loader_normal,  
            val_loader=test_loader,
            verbose=False
        )
        
        # Save the results
        val_auc = np.mean(val_auc_scores[-10:])
        print(f"=> Validation AUC: {val_auc:.4f}")
        print("-------------------------")
        results.append((config, val_auc))
        
        # Update the best configuration
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_config = config
    
    print("Best configuration found:")
    print(best_config)
    print("Validation AUC:", best_val_auc)
    return best_config, results


def visualize_results(results):
    configs = [result[0] for result in results]
    val_aucs = [result[1] for result in results]
    
    params = {
        'Latent Dimension': [config['latent_dim'] for config in configs],
        'Learning Rate': [config['lr'] for config in configs],
        'Encoder Dropout': [config['enc_dropout'] for config in configs],
        'Convolution Kernel': [config['conv']['ker'] for config in configs],
        'Decoder Dropout': [config['dec_dropout'] for config in configs],
        'Batch Norm': [config['batch_norm'] for config in configs],
        'Max Noise': [config['max_noise'] for config in configs]
    }

    for param_name, param_values in params.items():
        plt.figure(figsize=(5, 4))
        plt.scatter(param_values, val_aucs, alpha=0.7, label="Validation AUC")
        if param_name == 'Learning Rate':
            plt.xscale('log')
        plt.xlabel(param_name)
        plt.ylabel('Validation AUC')
        plt.title(f'Validation AUC vs {param_name}')
        plt.grid(True)
        plt.show()       

        