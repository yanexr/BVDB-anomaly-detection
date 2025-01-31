import os
import pandas as pd
import numpy as np
import neurokit2 as nk
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset

def load_data(filtered=True):
    raw_data_path = os.path.join(os.path.dirname(__file__), 'raw_data.pkl')
    filtered_data_path = os.path.join(os.path.dirname(__file__), 'filtered_data.pkl')
    if filtered and os.path.exists(filtered_data_path):
        return pd.read_pickle(filtered_data_path)
    elif not filtered and os.path.exists(raw_data_path):
        return pd.read_pickle(raw_data_path)
    
    starting_point_path = os.path.join(os.path.dirname(__file__), 'PartA', 'starting_point', 'samples.csv')
    df_starting_point = pd.read_csv(starting_point_path, sep='\t')
    
    # Choose 'biosignals_raw/' or 'biosignals_filtered/'
    biosignals_dir = 'biosignals_filtered' if filtered else 'biosignals_raw'
    
    data = []
    # Load the data
    for i, row in df_starting_point.iterrows():
        # Load the biosignals
        file_path = os.path.join(os.path.dirname(__file__), 'PartA', biosignals_dir, row['subject_name'], row['sample_name'] + '_bio.csv')
        bio_signals = pd.read_csv(file_path, sep='\t')[['gsr', 'ecg', 'emg_trapezius']]
        # Load the temperature
        temp_file_path = os.path.join(os.path.dirname(__file__), 'PartA', 'temperature', row['subject_name'], row['sample_name'] + '_temp.csv')
        temperature = pd.read_csv(temp_file_path, sep='\t')['temperature']

        data.append({'subject_name': row['subject_name'], 'sample_name': row['sample_name'], 'bio_signals': bio_signals, 'temperature': temperature, 'class_id': row['class_id']})
    
    if filtered:
        pd.DataFrame(data).to_pickle(filtered_data_path)
    else:
        pd.DataFrame(data).to_pickle(raw_data_path)

    return pd.DataFrame(data)


def smooth(signal, sigma=60):
    '''
    Smooth the signal with a Gaussian kernel.
    '''
    return gaussian_filter1d(signal, sigma=sigma, mode='nearest')

def bandpass_filter(signal, lowcut=0.1, highcut=250, order=4):
    '''
    Bandpass filtering with a butterworth filter.
    For ECG [0.1, 250] Hz with order 3.
    For EMG [20, 250] Hz with order 4.
    '''
    return nk.signal_filter(signal, lowcut=lowcut, highcut=highcut, sampling_rate=512, method='butterworth', order=order)

def detrend(signal):
    '''
    Detrending by subtracting a fifth-degree polynomial least-squares fit.
    '''
    x = np.arange(len(signal))
    p = np.polynomial.Polynomial.fit(x, signal, deg=5)
    poly_fit = p(x)
    return signal - poly_fit

def z_normalize(signal, mean, std):
    '''
    Z-normalization.
    '''
    return (signal - mean) / std

def resample(signal, target_length):
    '''
    Downsample the signal to the target length.
    '''
    x = np.linspace(0, 1, len(signal))
    f = interp1d(x, signal)
    x_new = np.linspace(0, 1, int(target_length))
    return f(x_new)
    

def preprocess(data, emg_envelope=False, resample_length=2816):
    '''
    Preprocess the data by applying the following steps:
    1. Smooth the GSR signal
    2. Detrend and bandpass filter the ECG signal
    3. Bandpass filter the EMG signal (optionally calculate the envelope)
    4. Resample the signals
    5. Z-normalize the signals using normal mean and std
    '''
    new_data = data.copy(deep=True)
    for index, row in new_data.iterrows():
        bio_signals = row['bio_signals'].copy(deep=True)
        bio_signals['gsr'] = smooth(bio_signals['gsr'])
        bio_signals['ecg'] = detrend(bandpass_filter(bio_signals['ecg'], lowcut=0.1, highcut=250, order=3))
        bio_signals['emg_trapezius'] = bandpass_filter(bio_signals['emg_trapezius'], lowcut=20, highcut=250, order=4)
        if emg_envelope:
            bio_signals['emg_trapezius'] = smooth(np.abs(bio_signals['emg_trapezius']), sigma=20)
        new_data.at[index, 'bio_signals'] = bio_signals

    if resample_length != 2816:
        for index, row in new_data.iterrows():
            # Create new resampled signals
            resampled_gsr = resample(row['bio_signals']['gsr'].values, resample_length)
            resampled_ecg = resample(row['bio_signals']['ecg'].values, resample_length)
            resampled_emg = resample(row['bio_signals']['emg_trapezius'].values, resample_length)
            
            # Create a new DataFrame for bio_signals with the resampled data
            bio_signals = pd.DataFrame({
                'gsr': resampled_gsr,
                'ecg': resampled_ecg,
                'emg_trapezius': resampled_emg
            })
            
            # Assign the new bio_signals DataFrame back to new_data
            new_data.at[index, 'bio_signals'] = bio_signals


    # Calculate the mean and standard deviation of the normal class
    gsr_values = np.concatenate(new_data.query('class_id == 0')['bio_signals'].apply(lambda signals: np.array(signals['gsr'])).to_list())
    ecg_values = np.concatenate(new_data.query('class_id == 0')['bio_signals'].apply(lambda signals: np.array(signals['ecg'])).to_list())
    emg_values = np.concatenate(new_data.query('class_id == 0')['bio_signals'].apply(lambda signals: np.array(signals['emg_trapezius'])).to_list())
    mean_normal_gsr = gsr_values.mean()
    std_normal_gsr = gsr_values.std()
    mean_normal_ecg = ecg_values.mean()
    std_normal_ecg = ecg_values.std()
    mean_normal_emg = emg_values.mean()
    std_normal_emg = emg_values.std()

    # Z-normalization
    for index, row in new_data.iterrows():
        row['bio_signals']['gsr'] = z_normalize(row['bio_signals']['gsr'], mean_normal_gsr, std_normal_gsr)
        row['bio_signals']['ecg'] = z_normalize(row['bio_signals']['ecg'], mean_normal_ecg, std_normal_ecg)
        row['bio_signals']['emg_trapezius'] = z_normalize(row['bio_signals']['emg_trapezius'], mean_normal_emg, std_normal_emg)
    
    return new_data


class Subjects:
    '''
    Provides a split of subjects into training and validation sets, as [proposed by Stefanos Gkikas](https://www.nit.ovgu.de/nit_media/Bilder/Dokumente/BIOVID_Dokumente/BioVid_HoldOutEval_Proposal.pdf), for a less computationally expensive evaluation method compared to LOSO-CV.

    Static Variables:
    ----------------
    all : list
        all unique subjects in the dataset (87 subjects)
    val : list
        subjects to be used for validation (26 subjects/ 30%)
    train : list
        subjects to be used for training (61 subjects/ 70%)
    '''
    all = load_data()['subject_name'].unique()

    val = ['100914_m_39','101114_w_37','082315_w_60','083114_w_55','083109_m_60','072514_m_27','080309_m_29','112016_m_25','112310_m_20','092813_w_24','112809_w_23','112909_w_20','071313_m_41','101309_m_48','101609_m_36','091809_w_43','102214_w_36','102316_w_50','112009_w_43','101814_m_58','101908_m_61','102309_m_61','112209_m_51','112610_w_60','112914_w_51','120514_w_56']

    train = np.setdiff1d(all, val)



def prepare_data(df, signals=['gsr', 'ecg', 'emg_trapezius'], classes=[0, 4], train_subjects=None, test_subjects=None, leave_one_out_subject=None, leave_one_in_subject=None, mp_filter=0):

    if mp_filter>0:
        # Filter out samples with high anomaly scores according to the matrix profile
        mp_results = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'mp_results.csv'))
        mp_results = mp_results[mp_results['pain'] == 0]
        mp_results = mp_results.sort_values(by='anomaly_score', ascending=False)
        mp_results = mp_results.head(int(len(mp_results)*mp_filter))
        sample_names_to_exclude = mp_results['sample_name'].values

    new_df = df[['sample_name', 'class_id']].copy()
    def get_signals(row):
        return row['bio_signals'][signals]
    new_df['signals'] = df.apply(get_signals, axis=1)

    # include a column with the anomaly scores according to the matrix profile
    mp_results = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'mp_results.csv'))
    new_df = new_df.merge(mp_results[['sample_name', 'anomaly_score']], on='sample_name', how='left')

    # Filter classes
    new_df = new_df[new_df['class_id'].isin(classes)]
    new_df['class_id'] = new_df['class_id'].apply(lambda x: 1 if x > 0 else 0)

    # create train and test sets
    if leave_one_out_subject:
        train_data = new_df[new_df['sample_name'].str.startswith(leave_one_out_subject) == False]
        test_data = new_df[new_df['sample_name'].str.startswith(leave_one_out_subject)]

    elif leave_one_in_subject:
        train_data = new_df[new_df['sample_name'].str.startswith(leave_one_in_subject)]
        test_data = new_df[new_df['sample_name'].str.startswith(leave_one_in_subject) == False]

    elif train_subjects is not None and test_subjects is not None:
        train_data = new_df[new_df['sample_name'].str.startswith(tuple(train_subjects))]
        test_data = new_df[new_df['sample_name'].str.startswith(tuple(test_subjects))]

    else:
        raise ValueError('Either leave_one_out_subject or train_subjects and validation_subjects must be provided')
    
    if mp_filter>0:
        # Exclude samples with high anomaly scores
        train_data = train_data[~train_data['sample_name'].isin(sample_names_to_exclude)]
    
    return train_data, test_data


class BVDBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        signals_df = row['signals']
        signals = torch.tensor(signals_df.values, dtype=torch.float32)
        label = torch.tensor(row['class_id'], dtype=torch.float32)
        return signals, label


class ContextBVDBDataset(Dataset):
    '''
    Dataset that includes normal samples from the same subject as context.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with the data
    k : int
        Number of normal samples to include as context (maximum 10)
    '''
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        signals_df = row['signals']
        main_signals = torch.tensor(signals_df.values, dtype=torch.float32)
        label = torch.tensor(row['class_id'], dtype=torch.float32)

        subject_name = row['sample_name'][:11]
        same_subject_data = self.data[self.data['sample_name'].str.startswith(subject_name)]
        
        # select k normal samples with the lowest anomaly scores as context, excluding the main sample
        context_samples = same_subject_data[(same_subject_data['class_id'] == 0) & (same_subject_data['sample_name'] != row['sample_name'])].sort_values(by='anomaly_score', ascending=True)
        context_samples = context_samples.head(10)
        context_samples = context_samples.sample(n=self.k)

        context_signals_list = []
        for _, context_row in context_samples.iterrows():
            context_signals_df = context_row['signals']
            context_signals = torch.tensor(context_signals_df.values, dtype=torch.float32)
            context_signals_list.append(context_signals)

        # Concatenate context signals
        context_signals = torch.stack(context_signals_list, dim=0)
        # Concatenate main signals with context signals
        combined_signals = torch.cat([main_signals.unsqueeze(0), context_signals], dim=0)
        combined_signals = combined_signals.permute(1, 0, 2)  # Change shape to (seq, k+1, channels)
        return combined_signals, label



def get_dataloader(data, batch_size, shuffle, k=0, num_workers=0):
    if k > 0:
        dataset = ContextBVDBDataset(data, k)
    else:
        dataset = BVDBDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        


def get_dataloaders_subject_specific(df, signal, batch_size, subject, mp_filter=5, fold=0):
    data, _ = prepare_data(df, signals=[signal], train_subjects=[subject], test_subjects=[subject], mp_filter=0)
    normal_data = data[data['class_id'] == 0]
    anomaly_data = data[data['class_id'] == 1]
    
    # test data contains 5 samples
    test_data_normal = normal_data.iloc[fold*5:(fold+1)*5]
    test_data_anomaly = anomaly_data.iloc[fold*5:(fold+1)*5]
    test_data = pd.concat([test_data_normal, test_data_anomaly])

    # train data contains all samples except the ones in the test data
    train_data = data[~data['sample_name'].isin(test_data_normal['sample_name'])]
    train_data_normal = train_data[train_data['class_id'] != 4]

    # filter out samples with high anomaly scores according to the matrix profile
    train_data_normal = train_data_normal.sort_values(by='anomaly_score', ascending=False)
    train_data_normal = train_data_normal.iloc[mp_filter:]

    test_loader = DataLoader(BVDBDataset(test_data), batch_size=10, shuffle=False)
    train_loader = DataLoader(BVDBDataset(train_data), batch_size=batch_size, shuffle=True)
    train_normal_loader = DataLoader(BVDBDataset(train_data_normal), batch_size=batch_size, shuffle=True)

    return train_loader, train_normal_loader, test_loader







