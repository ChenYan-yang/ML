import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import myLSTM, myTransformer, Trendformer
from utils import scale_features, sliding_window, TimeSeriesDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

def plot_multiple_predictions(pred_list, ground_truth, model_name, future_len=90 ,sample_idx=0):
    plt.figure(figsize=(12, 6))
    colors = ['red', 'green', 'orange', 'purple', 'cyan']

    # 绘制每次预测结果
    for i, pred in enumerate(pred_list):
        plt.plot(pred[sample_idx], label=f'Prediction {i+1}', color=colors[i % len(colors)], alpha=0.6)

    # 绘制 Ground Truth
    plt.plot(ground_truth[sample_idx], label='Ground Truth', color='black', linewidth=2)

    plt.title(f'{model_name} - 5 Runs Prediction vs Ground Truth')
    plt.xlabel('Days into the Future')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../result/{model_name}_{future_len}days.png')
    plt.close()

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, scaler, criterion):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            preds.append(pred.cpu().numpy())
            trues.append(Y_batch.cpu().numpy())
            total_loss += loss.item() * X_batch.size(0)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    preds_unscaled = scaler['target'].inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_unscaled = scaler['target'].inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

    mse = mean_squared_error(trues_unscaled.flatten(), preds_unscaled.flatten())
    mae = mean_absolute_error(trues_unscaled.flatten(), preds_unscaled.flatten())
    return mse, mae, preds_unscaled, trues_unscaled, total_loss / len(dataloader.dataset)


def run(model_name, X_train, Y_train, X_test, Y_test, scaler, epochs=100, batch_size=64, lr=0.001):
    if model_name == 'LSTM':
        model = myLSTM(X_train.shape[2], Y_train.shape[1]).to(device)
    elif model_name == 'Transformer':
        model = myTransformer(X_train.shape[2], Y_train.shape[1]).to(device)
    else:
        model = Trendformer(X_train.shape[2], Y_train.shape[1]).to(device)
    
    train_dataset = TimeSeriesDataset(X_train, Y_train)
    test_dataset = TimeSeriesDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in tqdm(range(epochs), desc='Epochs'):
        train_loss = train(model, train_loader, optimizer, criterion)
        mse, mae, preds_unscaled, trues_unscaled, test_loss = evaluate(model, test_loader, scaler, criterion)

    mse, mae, preds_unscaled, trues_unscaled, test_loss = evaluate(model, test_loader, scaler, criterion)
    print(f"{model_name} Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    return mse, mae, preds_unscaled, trues_unscaled

def run_evaluate(model_name, future_len, X_train, Y_train, X_test, Y_test, scaler, epochs, lr, filepath='../result/evaluate.txt'):
    # 存储5次结果
    pred_runs = []
    mses, maes = [], []
    for i in range(5):
        print('-'*30 + f'running {i}-th round {model_name} for predicting {future_len} days' + '-'*30)
        mse, mae, preds_unscaled, trues_unscaled = run(model_name, X_train, Y_train, X_test, Y_test, scaler, epochs=epochs, lr=lr)
        pred_runs.append(preds_unscaled)
        mses.append(mse)
        maes.append(mae)
    print(f"{model_name} - 5 Run Avg MSE: {np.mean(mses):.2f}, Std: {np.std(mses):.2f}")
    print(f"{model_name} - 5 Run Avg MAE: {np.mean(maes):.2f}, Std: {np.std(maes):.2f}")    


    with open(filepath, 'a') as f:
        f.write('-'*10 + f'{model_name } for {future_len} days' + '-'*10 + '\n')
        f.write(f'Avg MSE: {np.mean(mses):.2f}, Std: {np.std(mses):.2f}' + '\n')
        f.write(f'Avg MAE: {np.mean(maes):.2f}, Std: {np.std(maes):.2f}' + '\n')
        f.write('' + '\n')


    # 绘图对比
    os.makedirs('../result', exist_ok=True)
    plot_multiple_predictions(pred_runs, trues_unscaled, model_name=model_name, future_len=future_len, sample_idx=0)

if __name__ == "__main__":
    # 读取数据
    # 读取数据
    train_df = pd.read_csv('../Data/daily_train.csv', index_col='DateTime')
    test_df = pd.read_csv('../Data/daily_test.csv', index_col='DateTime')

    # 转float
    for col in train_df.columns:
        train_df[col] = train_df[col].astype('float32')
        test_df[col] = test_df[col].astype('float32')

    # 特征缩放
    train_scaled, test_scaled, scaler = scale_features(train_df, test_df)

    # 构造滑动窗口
    past_len_short, future_len_short = 90, 90
    past_len_long, future_len_long = 90, 365

    # 短期预测
    X_train_short, Y_train_short = sliding_window(train_scaled, past_len_short, future_len_short)
    X_test_short, Y_test_short = sliding_window(test_scaled, past_len_short, future_len_short)

    # 长期预测
    X_train_long, Y_train_long = sliding_window(train_scaled, past_len_long, future_len_long)
    X_test_long, Y_test_long = sliding_window(test_scaled, past_len_long, future_len_long)

    run_evaluate('LSTM', 90, X_train_short, Y_train_short, X_test_short, Y_test_short, scaler, epochs=40, lr=2e-5)
    run_evaluate('Transformer', 90, X_train_short, Y_train_short, X_test_short, Y_test_short, scaler, epochs=50, lr=2e-5)
    run_evaluate('Trendformer', 90, X_train_short, Y_train_short, X_test_short, Y_test_short, scaler, epochs=50, lr=2e-5)
    run_evaluate('LSTM', 365, X_train_long, Y_train_long, X_test_long, Y_test_long, scaler, epochs=75, lr=2e-5)
    run_evaluate('Transformer', 365, X_train_long, Y_train_long, X_test_long, Y_test_long, scaler, epochs=40, lr=1e-5)
    run_evaluate('Trendformer', 365, X_train_long, Y_train_long, X_test_long, Y_test_long, scaler, epochs=45, lr=1e-5)
