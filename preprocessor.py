import numpy as np
import pandas as pd
import os
import wfdb


# 60 nodes representing 60 seconds
def preprocess(name_list, file_path, fs):
    X, y = [], []
    for name in name_list:
        ecg = wfdb.rdsamp('/'.join([file_path, name]))[0]
        ecg = ecg / ecg.max()    # normalization
        anno_info = wfdb.rdann('/'.join([file_path, name]), extension="apn")
        sample = anno_info.sample
        y.append(anno_info.symbol)
        for i in range(len(sample)):
            if i == 0:
                seg = ecg[: sample[i + 1]]
            elif i == len(sample) - 1:
                seg = ecg[sample[i - 1]: sample[i]]
            else:
                seg = ecg[-30 * fs + sample[i]: 30 * fs + sample[i]]
            seg_2d = seg.reshape((60, fs)).transpose()
            X.append(seg_2d)
    X = np.stack(X, axis=0)
    y = pd.factorize(np.concatenate(y))[0]
    return X, y

fs = 100
file_path = 'data/apnea-ecg'
name_train = list(set([n[:3] for n in os.listdir(file_path) if n[0] in ['a', 'b', 'c'] and n.split('.')[0][-1].isdigit()]))
name_test = list(set([n[:3] for n in os.listdir(file_path) if n[0] == 'x' and n.split('.')[0][-1].isdigit()]))
X_train, y_train = preprocess(name_train, file_path, fs)
X_test, y_test = preprocess(name_test, file_path, fs)


os.listdir('data/apnea-ecg')
