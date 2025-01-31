import numpy as np
import stumpy

def stump_for_subjects(df, subjects, class_id, signal, m, tb_indices=None, normal_only=False):
    """
    Compute the STUMP matrix profiles of the normal and anomaly samples for each subject individually.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    subjects : list
        The list of subjects.
    class_id : int
        The class ID of the anomaly samples.
    signal : str
        The signal to use.
    m : int
        The window size.
    tb_indices : list or bool, optional
        The indices of the normal samples of each subject to use as t_b. If None, all normal samples are used as t_b. If False, the self join is performed.
    normal_only : bool, optional
        If True, only the matrix profiles of the normal samples are computed.

    Returns
    -------
    subjects_mps_0 : list
        The matrix profiles of the normal samples for each subject.
    subjects_mps_1 : list
        The matrix profiles of the anomaly samples for each subject.
    scores_0 : list
        The scores of the normal samples for each subject.
    scores_1 : list
        The scores of the anomaly samples for each subject.
    """
    ignore_trivial = True if tb_indices is False else False

    subjects_mps_0 = []
    subjects_mps_1 = []
    
    for s, subject in enumerate(subjects):
        mps_0 = []
        mps_1 = []
        normal_data = df.query("class_id == 0 and subject_name == @subject")['bio_signals'].apply(lambda signals: signals.get(signal).values)
        anomaly_data = df.query("class_id == @class_id and subject_name == @subject")['bio_signals'].apply(lambda signals: signals.get(signal).values)

        # calculate the matrix profiles for the normal samples
        for i, t_a in enumerate(normal_data):
            t_b = []
            if tb_indices == False:
                # self join
                t_b = None
            elif tb_indices is None:
                # t_b is the concatenation of all normal samples except the current one
                for j, normal_j in enumerate(normal_data):
                    if i != j:
                        t_b = np.concatenate([t_b, [np.nan], normal_j])
            else:
                # t_b is only a specific normal sample
                if i == tb_indices[s][0] and len(tb_indices[s]) == 1:
                    continue
                # t_b is the concatenation of only some normal samples
                for tb_index in tb_indices[s]:
                    if i != tb_index:
                        t_b = np.concatenate([t_b, [np.nan], normal_data.iloc[tb_index]])
            mp = stumpy.stump(T_A=t_a, m=m, T_B=t_b, ignore_trivial=ignore_trivial)[:, 0]
            mps_0.append(mp)
        subjects_mps_0.append(mps_0)

        if normal_only:
            subjects_mps_1.append([[0]])
            continue

        # calculate the matrix profiles for the anomaly samples
        for i, t_a in enumerate(anomaly_data):
            t_b = []
            if tb_indices == False:
                # self join
                t_b = None
            elif tb_indices is None:
                # t_b is the concatenation of all normal samples except the current one
                for j, normal_j in enumerate(normal_data):
                    if i != j:
                        t_b = np.concatenate([t_b, [np.nan], normal_j])
            else:
                # t_b is only a specific normal sample
                if i == tb_indices[s][0] and len(tb_indices[s]) == 1:
                    continue
                # t_b is the concatenation of only some normal samples
                for tb_index in tb_indices[s]:
                    if i != tb_index:
                        t_b = np.concatenate([t_b, [np.nan], normal_data.iloc[tb_index]])
            mp = stumpy.stump(T_A=t_a, m=m, T_B=t_b, ignore_trivial=ignore_trivial)[:, 0]
            mps_1.append(mp)
        subjects_mps_1.append(mps_1)

    scores_0, scores_1 = score_mps(subjects_mps_0), score_mps(subjects_mps_1)
    return subjects_mps_0, subjects_mps_1, scores_0, scores_1




def mpdist_for_subjects(df, subjects, class_id, signal, m):
    """
    Compute the matrix profile distances (MPdist) of the normal and anomaly samples for each subject individually.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    subjects : list
        The list of subjects.
    class_id : int
        The class ID of the anomaly samples.
    signal : str
        The signal to use.
    m : int
        The window size.

    Returns
    -------
    subjects_dists_0 : list
        The matrix profile distances/ scores of the normal samples for each subject.
    subjects_dists_1 : list
        The matrix profile distances/ scores of the anomaly samples for each subject.
    """

    subjects_dists_0 = []
    subjects_dists_1 = []
    
    for subject in subjects:
        dists_0 = []
        dists_1 = []
        normal_data = df.query("class_id == 0 and subject_name == @subject")['bio_signals'].apply(lambda signals: signals.get(signal).values)
        anomaly_data = df.query("class_id == @class_id and subject_name == @subject")['bio_signals'].apply(lambda signals: signals.get(signal).values)

        # calculate the matrix profiles for the normal samples
        for i, t_a in enumerate(normal_data):
            t_b = []
            for j, normal_j in enumerate(normal_data):
                if i != j:
                    t_b = np.concatenate([t_b, [np.nan], normal_j])
            dist = stumpy.mpdist(T_A=t_a, T_B=t_b, m=m)
            dists_0.append(dist)

        # calculate the matrix profiles for the anomaly samples
        for i, t_a in enumerate(anomaly_data):
            t_b = []
            for j, normal_j in enumerate(normal_data):
                if i != j:
                    t_b = np.concatenate([t_b, [np.nan], normal_j])
            dist = stumpy.mpdist(T_A=t_a, T_B=t_b, m=m)
            dists_1.append(dist)
        
        subjects_dists_0.append(dists_0)
        subjects_dists_1.append(dists_1)

    return subjects_dists_0, subjects_dists_1




def multi_stump_for_subjects(df, subjects, class_id, signals, ms, tb_indices=None):
    """
    Compute STUMP matrix profiles of the normal and anomaly samples for each subject and each signal individually. The matrix profiles of different signals are then summed into one matrix profile.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    subjects : list
        The list of subjects.
    class_id : int
        The class ID of the anomaly samples.
    signals : list
        The list of signals to use.
    ms : list
        The list of window sizes of each signal.
    tb_indices : list, optional
        For each signal, a list that contains the indices of the normal samples of each subject to use as t_b. If None, all normal samples are used.

    Returns
    -------
    subjects_multi_mps_0 : list
        The summed matrix profiles of the normal samples for each subject.
    subjects_multi_mps_1 : list
        The summed matrix profiles of the anomaly samples for each subject.
    scores_0 : list
        The scores of the normal samples for each subject.
    scores_1 : list
        The scores of the anomaly samples for each subject.
    """
    signal_mps = []

    # calculate the matrix profiles for each signal
    for i in range(len(signals)):
        s_mps_0, s_mps_1, _, _ = stump_for_subjects(df, subjects, class_id, signals[i], ms[i], tb_indices=None if tb_indices is None else tb_indices[i])
        signal_mps.append((s_mps_0, s_mps_1))

    for i, (subjects_mps_0, subjects_mps_1) in enumerate(signal_mps): # for each signal
        for (mps_0, mps_1) in zip(subjects_mps_0, subjects_mps_1): # for each subject
            mean = np.mean(mps_0)
            std = np.std(mps_0)
            for k, (mp_0, mp_1) in enumerate(zip(mps_0, mps_1)):
                # make both same length
                length_diff = ms[i] - min(ms)
                if length_diff > 0:
                    # continue the matrix profile with the mean of the matrix profile
                    mp_0 = np.concatenate([mp_0, np.full(length_diff, np.mean(mp_0))])
                    mp_1 = np.concatenate([mp_1, np.full(length_diff, np.mean(mp_1))])
                # z-normalize
                mps_0[k] = (mp_0 - mean) / std
                mps_1[k] = (mp_1 - mean) / std

    # sum the matrix profiles
    subjects_multi_mps_0 = []
    subjects_multi_mps_1 = []
    for i in range(len(signal_mps[0][0])): # for each subject
        multi_mps_0 = []
        multi_mps_1 = []
        for j in range(len(subjects_mps_0[0])): # for each matrix profile
            multi_mps_0.append(np.sum([signal_mps[s][0][i][j] for s in range(len(signal_mps))], axis=0))
            multi_mps_1.append(np.sum([signal_mps[s][1][i][j] for s in range(len(signal_mps))], axis=0))
        
        subjects_multi_mps_0.append(multi_mps_0)
        subjects_multi_mps_1.append(multi_mps_1)

    scores_0, scores_1 = score_mps(subjects_multi_mps_0), score_mps(subjects_multi_mps_1)
    return subjects_multi_mps_0, subjects_multi_mps_1, scores_0, scores_1




def score_mps(subjects_mps):
    """
    Transforms matrix profiles into scores by taking the 70th percentile of each matrix profile.
    """
    subjects_scores = [
        [np.percentile(mp_0, 70) for mp_0 in mps_0]
        for mps_0 in subjects_mps
    ]

    # Or as in mpdist: take the kth smallest number as the reported score,
    # k is equal to 5 percent of T_A and T_B.
    #k = int(0.05 * 2816)
        
    return subjects_scores