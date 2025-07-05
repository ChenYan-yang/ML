import numpy as np
import pandas as pd
from datetime import datetime

def preprocess_data(filepath, savepath):
    # 加载原始数据
    column_names = [
        'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]

    if 'train' in filepath:
        df = pd.read_csv(
        filepath,
        header = 0, 
        index_col = ['DateTime'],
        low_memory = False
        )

    else:
        df = pd.read_csv(
            filepath,
            header = None, # test.csv中没有表头
            names = column_names,
            index_col = ['DateTime'],
            low_memory = False
        )

    df.index = pd.to_datetime(df.index)

    # 将缺失值标记，并将数据转换为float形式
    df.replace('?', np.nan, inplace=True)
    df = df.astype('float32')

    # 缺失值填充，用前后两天
    for m in [1440, 2880, -1440, -2880]:
        df = df.fillna(df.shift(m))

    # 添加剩余电量值
    sub_metering_remainder = (df['Global_active_power'] * 1000 / 60) - (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
    df['Sub_metering_remainder'] = sub_metering_remainder

    # 按天将数据聚合
    daily_data = pd.DataFrame()

    for col in ['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder']:
        daily_data[col] = df[col].resample("D").sum()

    for col in ['Voltage', 'Global_intensity']:
        daily_data[col] = df[col].resample("D").mean()

    for col in ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']:
        daily_data[col] = df[col].resample("D").first()

    # 保存数据
    daily_data.to_csv(savepath)

if __name__ == "__main__":
    preprocess_data('../Data/train.csv', '../Data/daily_train.csv')
    preprocess_data('../Data/test.csv', '../Data/daily_test.csv')