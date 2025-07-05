import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 数据标准化处理
def scale_features(train_df, test_df):
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # 定义特征列
    standard_features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_remainder']
    log_standard_features = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    minmax_features = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    scalers = {}
    
    # 标准化 Global_active_power（目标值）
    scalers['target'] = StandardScaler()
    train_scaled['Global_active_power'] = scalers['target'].fit_transform(train_df[['Global_active_power']])
    test_scaled['Global_active_power'] = scalers['target'].transform(test_df[['Global_active_power']])

    # 标准化普通数值特征
    scalers['standard'] = StandardScaler()
    train_scaled[standard_features] = scalers['standard'].fit_transform(train_df[standard_features])
    test_scaled[standard_features] = scalers['standard'].transform(test_df[standard_features])

    # 对数标准化特征
    log_train = np.log1p(train_df[log_standard_features])
    log_test = np.log1p(test_df[log_standard_features])
    scalers['log'] = StandardScaler()
    train_scaled[log_standard_features] = scalers['log'].fit_transform(log_train)
    test_scaled[log_standard_features] = scalers['log'].transform(log_test)

    # 最大-最小缩放特征
    scalers['minmax'] = MinMaxScaler()
    train_scaled[minmax_features] = scalers['minmax'].fit_transform(train_df[minmax_features])
    test_scaled[minmax_features] = scalers['minmax'].transform(test_df[minmax_features])

    return train_scaled, test_scaled, scalers


# 滑动窗口->构建训练集、测试集
def sliding_window(data, past_len, future_len):
    X, Y = [], []
    data = np.array(data)
    for i in range(len(data) - past_len - future_len + 1):
        X.append(data[i : i+past_len])  # past_len的特征序列
        Y.append(data[i+past_len : i+past_len+future_len, 0])   # feature_len的目标值
    return np.array(X), np.array(Y)


# 定义数据类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: numpy.ndarray, shape (num_samples, past_len, num_features)
        Y: numpy.ndarray, shape (num_samples, future_len)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]