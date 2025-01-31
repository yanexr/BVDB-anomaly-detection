import numpy as np
import neurokit2 as nk
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

def extract_features(data):
    features_list = []
    data = data.reset_index(drop=True)

    for index, row in data.iterrows():
        bio_signals = row['bio_signals']
        eda = eda_features(bio_signals['gsr'].to_numpy())
        ecg = ecg_features(bio_signals['ecg'].to_numpy())
        emg = emg_features(bio_signals['emg_trapezius'].to_numpy())
        
        features = {**eda, **ecg, **emg}
        features['subject_id'] = row['subject_name']
        features['class'] = row['class_id']

        print(f'\r{index + 1}/{len(data)}', end='')
        features_list.append(features)
    return pd.DataFrame(features_list)

def eda_features(signal):
    '''
    - Max: maximum
    - Min: minimum
    - ArgMax: argument/ index of the maximum
    - ArgMin: argument/ index of the minimum
    - Mean: mean 
    - NormMean: normalized mean
    - StD: standard deviation
    - NormStD: normalized standard deviation
    - Var: variance
    - NormVar: normalized variance
    - Range: range (max - min)
    - DiffStartEnd: difference between the first and the last value
    - IqR: interquartile range
    - MeanPhasic: mean of the phasic component
    - StDPhasic: standard deviation of the phasic component
    - MinPhasic: minimum of the phasic component
    - MeanTonic: mean of the tonic component
    - StDTonic: standard deviation of the tonic component
    - RangeTonic: range of the tonic component (max - min)
    - Skew: skewness (the asymmetry of the distribution)
    - Kurtosis: kurtosis (the tailedness of the distribution)
    - Max1D: maximum of the first derivative
    - Min1D: minimum of the first derivative
    '''
    return {
        'eda_Max': max_value(signal),
        'eda_Min': min_value(signal),
        'eda_ArgMax': argmax(signal),
        'eda_ArgMin': argmin(signal),
        'eda_Mean': mean_value(signal),
        'eda_NormMean': norm_mean(signal),
        'eda_StD': std(signal),
        'eda_NormStD': norm_std(signal),
        'eda_Var': var(signal),
        'eda_NormVar': norm_var(signal),
        'eda_Range': signal_range(signal),
        'eda_DiffStartEnd': diff_start_end(signal),
        'eda_IqR': iqr(signal),
        'eda_MeanPhasic': mean_phasic(signal),
        'eda_StDPhasic': std_phasic(signal),
        'eda_MinPhasic': min_phasic(signal),
        'eda_MeanTonic': mean_tonic(signal),
        'eda_StDTonic': std_tonic(signal),
        'eda_RangeTonic': range_tonic(signal),
        'eda_Skew': skewness(signal),
        'eda_Kurtosis': kurtosis(signal),
        'eda_Max1D': max_derivative(signal),
        'eda_Min1D': min_derivative(signal)
    }

def ecg_features(signal):
    '''
    - NumRPeaks: number of R-peaks
    - MIBIs: mean of the inter-beat intervals/ R-R intervals
    - SlopeRR: average slope of the linear regression of the R-R intervals
    - MeanRR: mean of the R-R intervals
    - StDRR: standard deviation of the R-R intervals
    - RMSSD: root mean square of successive differences (square root of the mean of the squares of differences between successive R-R intervals)
    - RyStD: standard deviation of R amplitudes
    - RyRange: range of R Amplitudes (max - min)
    - PTxStD: standard deviation of the P-T durations
    - PTxRMSSD: root mean square of successive differences of the P-T durations
    - PTyStD: standard deviation of the P-T amplitudes
    - PTyRMSSD: root mean square of successive differences of the P-T amplitudes
    '''
    return {
        'ecg_NumRPeaks': num_r_peaks(signal),
        'ecg_MIBIs': mean_ibi(signal),
        'ecg_SlopeRR': slope_rr(signal),
        'ecg_MeanRR': mean_rr(signal),
        'ecg_StDRR': std_rr(signal),
        'ecg_RMSSD': rmssd(signal),
        'ecg_RyStD': ry_std(signal),
        'ecg_RyRange': ry_range(signal),
        'ecg_PTxStD': ptx_std(signal),
        'ecg_PTxRMSSD': ptx_rmssd(signal),
        'ecg_PTyStD': ptystd(signal),
        'ecg_PTyRMSSD': ptyrmssd(signal)
    }

def emg_features(signal):
    '''
    - ZC: zero crossings
    - Var: variance
    - Max: maximum
    - IqR: interquartile range
    - MAV: mean absolute value
    - RMS: root mean square
    - StDStD: standard deviation of the standard deviation vector
    - ApEn: approximate entropy
    - ShannonEn: Shannon entropy
    '''
    return {
        'emg_ZC': zero_crossings(signal),
        'emg_Var': var(signal),
        'emg_Max': max_value(signal),
        'emg_IqR': iqr(signal),
        'emg_MAV': mav(signal),
        'emg_RMS': rms(signal),
        'emg_StDStD': std_std(signal),
        'emg_ApEn': approx_entropy(signal),
        'emg_ShannonEn': shannon_entropy(signal)
    }

# max
def max_value(signal):
    return np.max(signal)

# min
def min_value(signal):
    return np.min(signal)

# index of the maximum value
def argmax(signal):
    return np.argmax(signal)

# index of the minimum value
def argmin(signal):
    return np.argmin(signal)

# mean
def mean_value(signal):
    return np.mean(signal)

# normalized mean
def norm_mean(signal):
    range = np.max(signal) - np.min(signal)
    if range == 0:
        return 0
    return np.mean(signal) / range

# standard deviation
def std(signal):
    return np.std(signal)

# normalized standard deviation
def norm_std(signal):
    range = np.max(signal) - np.min(signal)
    if range == 0:
        return 0
    return np.std(signal) / range

# variance
def var(signal):
    return np.var(signal)

# normalized variance
def norm_var(signal):
    range = np.max(signal) - np.min(signal)
    if range == 0:
        return 0
    return np.var(signal) / range

# range
def signal_range(signal):
    return np.max(signal) - np.min(signal)

# difference between the first and last value
def diff_start_end(signal):
    return signal[-1] - signal[0]

# interquartile range
def iqr(signal):
    return np.percentile(signal, 75) - np.percentile(signal, 25)

# mean of the phasic component
def mean_phasic(signal):
    return np.mean(nk.eda_phasic(eda_signal=signal, sampling_rate=512)['EDA_Phasic'])

# standard deviation of the phasic component
def std_phasic(signal):
    return np.std(nk.eda_phasic(eda_signal=signal, sampling_rate=512)['EDA_Phasic'])

# minimum of the phasic component
def min_phasic(signal):
    return np.min(nk.eda_phasic(eda_signal=signal, sampling_rate=512)['EDA_Phasic'])

# mean of the tonic component
def mean_tonic(signal):
    return np.mean(nk.eda_phasic(eda_signal=signal, sampling_rate=512)['EDA_Tonic'])

# standard deviation of the tonic component
def std_tonic(signal):
    return np.std(nk.eda_phasic(eda_signal=signal, sampling_rate=512)['EDA_Tonic'])

# range tonic
def range_tonic(signal):
    tonic = nk.eda_phasic(eda_signal=signal, sampling_rate=512)['EDA_Tonic']
    return np.max(tonic) - np.min(tonic)

# skewness (the asymmetry of the distribution)
def skewness(signal):
    return stats.skew(signal)

# kurtosis (the tailedness of the distribution)
def kurtosis(signal):
    return stats.kurtosis(signal)

# maximum of the first derivative
def max_derivative(signal):
    return np.max(np.diff(signal))

# minimum of the first derivative
def min_derivative(signal):
    return np.min(np.diff(signal))

##################################################

# number of R peaks
def num_r_peaks(signal):
    return len(nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks'])

# mean of the inter-beat intervals/ R-R intervals
def mean_ibi(signal):
    r_peaks = nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks']
    rr_intervals = np.diff(r_peaks) / 512
    return np.mean(rr_intervals)

# average slope of the linear regression of the R-R intervals
def slope_rr(signal):
    r_peaks = nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks']
    rr_intervals = np.diff(r_peaks) / 512
    x = np.cumsum(rr_intervals).reshape(-1, 1)
    y = rr_intervals.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    return reg.coef_[0][0]

# mean of the R-R intervals
def mean_rr(signal):
    r_peaks = nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks']
    return nk.hrv_time(peaks=r_peaks, sampling_rate=512)['HRV_MeanNN'][0]

# standard deviation of the R-R intervals
def std_rr(signal):
    r_peaks = nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks']
    return nk.hrv_time(peaks=r_peaks, sampling_rate=512)['HRV_SDNN'][0]

# square root of the mean of the squares of differences between successive R-R intervals
def rmssd(signal):
    r_peaks = nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks']
    return nk.hrv_time(peaks=r_peaks, sampling_rate=512)['HRV_RMSSD'][0]

# standard deviation of R Amplitudes
def ry_std(signal):
    r_peaks = nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks']
    amplitudes = [signal[x] for x in r_peaks]
    return np.std(amplitudes)

# range of R Amplitudes
def ry_range(signal):
    r_peaks = nk.ecg_findpeaks(signal, sampling_rate=512)['ECG_R_Peaks']
    amplitudes = [signal[x] for x in r_peaks]
    return np.max(amplitudes) - np.min(amplitudes)

# P-T interval durations
def ptx_durations(signal):
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=512)
        p_peaks = info['ECG_P_Peaks']
        t_peaks = info['ECG_T_Peaks']
    except ValueError as e:
        print("Warning: P-T interval durations failed to compute")
        return [0]

    if p_peaks is None or t_peaks is None:
        return [0]

    durations = []
    for p, t in zip(p_peaks, t_peaks):
        if np.isnan(p) or np.isnan(t):
            continue
        durations.append(t - p)
    if len(durations) == 0:
        return [0]
    return [d for d in durations if not np.isnan(d)]

# standard deviation of the P-T interval durations
def ptx_std(signal):
    durations = ptx_durations(signal)
    return np.std(durations)

# root mean square of successive differences of PT interval durations
def ptx_rmssd(signal):
    durations = ptx_durations(signal)
    if len(durations) < 2:
        return np.nan
    return np.sqrt(np.mean(np.diff(durations) ** 2))

# P-T amplitude differences
def pty_differences(signal):
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=512)
        p_peaks = info['ECG_P_Peaks']
        t_peaks = info['ECG_T_Peaks']
    except ValueError as e:
        print("Warning: P-T amplitude differences failed to compute")
        return [0]

    if p_peaks is None or t_peaks is None:
        return [0]

    amplitude_diffs = []
    for p, t in zip(p_peaks, t_peaks):
        if np.isnan(p) or np.isnan(t):
            continue
        amplitude_diffs.append(signal[t] - signal[p])
    if len(amplitude_diffs) == 0:
        return [0]
    return [d for d in amplitude_diffs if not np.isnan(d)]

# standard deviation of the P-T amplitudes
def ptystd(signal):
    amplitudes = pty_differences(signal)
    return np.std(amplitudes)

# root mean square of successive differences of P-T amplitudes
def ptyrmssd(signal):
    amplitudes = pty_differences(signal)
    if len(amplitudes) < 2:
        return np.nan
    return np.sqrt(np.mean(np.diff(amplitudes) ** 2))

##################################################

# zero crossings
def zero_crossings(signal):
    zc = 0
    for i in range(1, len(signal)):
        if signal[i] * signal[i-1] < 0:
            zc += 1
    return zc

# mean absolute value
def mav(signal):
    return np.mean(np.abs(signal))

# root mean square
def rms(signal):
    return np.sqrt(np.mean(signal ** 2))

# standard deviation of standard deviation vector
def std_std(signal):
    segments = np.array_split(signal, 10)
    stds = [np.std(segment) for segment in segments]
    return np.std(stds)

# approximate entropy ApEn
def approx_entropy(signal):
    apen, info = nk.entropy_approximate(signal, delay=1, dimension=2, tolerance='sd')
    return apen

# shannon entropy
def shannon_entropy(signal):
    signal = pd.cut(signal, bins=100, labels=False)
    shanen, info = nk.entropy_shannon(signal, base=2)
    return shanen