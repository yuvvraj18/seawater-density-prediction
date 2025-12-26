import os
import time
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns


class Config:
    # data parameter
    input_columns = [0, 1, 2, 3]
    output_columns = [4]

    # network parameter
    input_size = 4
    output_size = 1
    hidden_size = [16, 32, 32]
    batch_size = 500

    test_data_path = 'D:/MatlabWork/seawater/Data density/2022a/'
    pre_model_name = '68-model.pth'  # 加载已有模型名字
    model_name = '-model.pth'  # model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_save_path = './checkpoint/'


class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=config.hidden_size[0], kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=config.hidden_size[0], out_channels=config.hidden_size[1], kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=config.hidden_size[1],out_features=config.hidden_size[2])
        self.fc2 = nn.Linear(in_features=config.hidden_size[2], out_features=config.output_size)
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Tanh()
        self.f3 = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.f1(self.conv1(x)))

        x = self.pool(self.f2(self.conv2(x)))

        x = x.view(-1, 32)

        x = self.f3(self.fc1(x))
        x = self.fc2(x)
        return x


def dict_scaler_para():
    X_dict = {'with_mean': True, 'with_std': True, 'copy': True, 'n_features_in_': 4,
              'mean_': np.array([-11.69089432, -21.48814215, 808.6271306,    6.62856361]),
              'var_': np.array([1.13293310e+03, 1.31229146e+04, 3.65833077e+05, 1.18371861e+01]),
              'scale_':np.array([ 33.65907159, 114.55529058, 604.84136515,   3.4405212 ])}
    Y_dict = {'with_mean': True, 'with_std': True, 'copy': True, 'n_features_in_': 1,
              'mean_': np.array([1030.57677758]), 'var_': np.array([13.6385557]), 'scale_':np.array([3.6930415242723713])}
    X_scaler = StandardScaler()
    for item in X_dict:
        setattr(X_scaler, item, X_dict[item])
    Y_scaler = StandardScaler()
    for item in Y_dict:
        setattr(Y_scaler, item, Y_dict[item])
    return X_scaler, Y_scaler

def predict(config, test_X):
    # 获取测试数据
    X_scaler, Y_scaler = dict_scaler_para()

    test_X = X_scaler.transform(test_X)
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    # device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    model = NeuralNetwork(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.pre_model_name))  # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()   # pytorch中，预测时要转换成预测模式
    for _data in test_loader:
        data_X = _data[0].to(device)
        data_X = data_X.unsqueeze(1)
        pred_X = model(data_X)
        # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)
    result = result.detach().cpu().numpy()
    result = Y_scaler.inverse_transform(result.reshape(-1, 1))
    return result  # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

if __name__ == '__main__':
    con = Config()

    device = torch.device('cpu')
    model = NeuralNetwork(con).to(device)
    model.load_state_dict(torch.load(con.model_save_path + con.pre_model_name))
    model.eval()
    X_scaler, Y_scaler = dict_scaler_para()

    # 53631,2 for lat and long
    pos = pd.read_excel('oceanposition.xlsx')
    pre = [0, 100, 200, 500, 1000, 2000]
    month = range(1, 13)


    # # The average annual sea water density is obtained
    # data = np.zeros([53631, 6])
    # for i in range(6):
    #     data_mean = np.zeros([53631])
    #     for m in month:
    #         test_X = np.concatenate((pos.values, pre[i]*np.ones([53631, 1]), m*np.ones([53631, 1])), axis=1)
    #         test_X = X_scaler.transform(test_X)
    #         num = test_X.shape[0]-test_X.shape[0] % con.batch_size
    #         test_X1 = torch.from_numpy(np.reshape(test_X[:num], (num, 1, con.input_size))).float()
    #         # test_set = TensorDataset(test_X)
    #         test_loader = DataLoader(TensorDataset(test_X1), batch_size=con.batch_size)
    #
    #         result = torch.Tensor().to(device)
    #         model.eval()
    #         for _data in test_loader:
    #             data_X = _data[0].to(device)
    #             # data_X = data_X.unsqueeze(1)
    #             pred_X = model(data_X)
    #             result = torch.cat((result, pred_X), dim=0)
    #         if test_X.shape[0] > num:
    #             test_X1 = torch.from_numpy(np.reshape(test_X[num:], (test_X.shape[0]-num, 1, con.input_size))).float()
    #             test_loader = DataLoader(TensorDataset(test_X1), batch_size=test_X.shape[0]-num)
    #             for _data in test_loader:
    #                 data_X = _data[0].to(device)
    #                 # data_X = data_X.unsqueeze(1)
    #                 pred_X = model(data_X)
    #                 result = torch.cat((result, pred_X), dim=0)
    #
    #         result = result.detach().cpu().numpy()
    #         result = Y_scaler.inverse_transform(result)
    #         result = np.squeeze(result)
    #         data_mean = data_mean + result
    #     data[:, i] = data_mean/12

    # # Obtain sea level monthly density
    # data = np.zeros([53631, 12])
    # for i in range(12):
    #     test_X = np.concatenate((pos.values, pre[0]*np.ones([53631, 1]), month[i]*np.ones([53631, 1])), axis=1)
    #     test_X = X_scaler.transform(test_X)
    #     num = test_X.shape[0]-test_X.shape[0] % con.batch_size
    #     test_X1 = torch.from_numpy(np.reshape(test_X[:num], (num, 1, con.input_size))).float()
    #     # test_set = TensorDataset(test_X)
    #     test_loader = DataLoader(TensorDataset(test_X1), batch_size=con.batch_size)
    #
    #     result = torch.Tensor().to(device)
    #     model.eval()  # pytorch中，预测时要转换成预测模式
    #     for _data in test_loader:
    #         data_X = _data[0].to(device)
    #         # data_X = data_X.unsqueeze(1)
    #         pred_X = model(data_X)
    #         # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
    #         # cur_pred = torch.squeeze(pred_X, dim=0)
    #         result = torch.cat((result, pred_X), dim=0)
    #     if test_X.shape[0] > num:
    #         test_X1 = torch.from_numpy(np.reshape(test_X[num:], (test_X.shape[0]-num, 1, con.input_size))).float()
    #         test_loader = DataLoader(TensorDataset(test_X1), batch_size=test_X.shape[0]-num)
    #         for _data in test_loader:
    #             data_X = _data[0].to(device)
    #             # data_X = data_X.unsqueeze(1)
    #             pred_X = model(data_X)
    #             # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
    #             # pred_X = torch.squeeze(pred_X, dim=0)
    #             result = torch.cat((result, pred_X), dim=0)
    #
    #     result = result.detach().cpu().numpy()
    #     result = Y_scaler.inverse_transform(result)
    #     result = np.squeeze(result)
    #     data[:, i] = result

    # Obtain depth density at different latitudes in different oceans
    lat = [-60, -30, 0, 30, 60]
    long = [-20, 75, -140]
    month = 6
    pre = np.arange(2201)
    pre = pre.reshape(-1, 1)
    data = np.zeros([2201, 13])
    for i in range(5):
        test_X = np.concatenate((lat[i]*np.ones([2201, 1]), long[0]*np.ones([2201, 1]), pre, month*np.ones([2201, 1])), axis=1)
        test_X = X_scaler.transform(test_X)
        num = test_X.shape[0]-test_X.shape[0] % con.batch_size
        test_X1 = torch.from_numpy(np.reshape(test_X[:num], (num, 1, con.input_size))).float()
        # test_set = TensorDataset(test_X)
        test_loader = DataLoader(TensorDataset(test_X1), batch_size=con.batch_size)

        result = torch.Tensor().to(device)
        model.eval()
        for _data in test_loader:
            data_X = _data[0].to(device)
            # data_X = data_X.unsqueeze(1)
            pred_X = model(data_X)
            result = torch.cat((result, pred_X), dim=0)
        if test_X.shape[0] > num:
            test_X1 = torch.from_numpy(np.reshape(test_X[num:], (test_X.shape[0]-num, 1, con.input_size))).float()
            test_loader = DataLoader(TensorDataset(test_X1), batch_size=test_X.shape[0]-num)
            for _data in test_loader:
                data_X = _data[0].to(device)
                # data_X = data_X.unsqueeze(1)
                pred_X = model(data_X)
                result = torch.cat((result, pred_X), dim=0)

        result = result.detach().cpu().numpy()
        result = Y_scaler.inverse_transform(result)
        result = np.squeeze(result)
        data[:, i] = result

    for i in range(3):
        test_X = np.concatenate((lat[i]*np.ones([2201, 1]), long[1]*np.ones([2201, 1]), pre, month*np.ones([2201, 1])), axis=1)
        test_X = X_scaler.transform(test_X)
        num = test_X.shape[0]-test_X.shape[0] % con.batch_size
        test_X1 = torch.from_numpy(np.reshape(test_X[:num], (num, 1, con.input_size))).float()
        # test_set = TensorDataset(test_X)
        test_loader = DataLoader(TensorDataset(test_X1), batch_size=con.batch_size)

        result = torch.Tensor().to(device)
        model.eval()
        for _data in test_loader:
            data_X = _data[0].to(device)
            # data_X = data_X.unsqueeze(1)
            pred_X = model(data_X)

            result = torch.cat((result, pred_X), dim=0)
        if test_X.shape[0] > num:
            test_X1 = torch.from_numpy(np.reshape(test_X[num:], (test_X.shape[0]-num, 1, con.input_size))).float()
            test_loader = DataLoader(TensorDataset(test_X1), batch_size=test_X.shape[0]-num)
            for _data in test_loader:
                data_X = _data[0].to(device)
                # data_X = data_X.unsqueeze(1)
                pred_X = model(data_X)

                result = torch.cat((result, pred_X), dim=0)

        result = result.detach().cpu().numpy()
        result = Y_scaler.inverse_transform(result)
        result = np.squeeze(result)
        # data[:, i] = result
        data[:, i+5] = result

    for i in range(5):
        test_X = np.concatenate((lat[i]*np.ones([2201, 1]), long[2]*np.ones([2201, 1]), pre, month*np.ones([2201, 1])), axis=1)
        test_X = X_scaler.transform(test_X)
        num = test_X.shape[0]-test_X.shape[0] % con.batch_size
        test_X1 = torch.from_numpy(np.reshape(test_X[:num], (num, 1, con.input_size))).float()
        # test_set = TensorDataset(test_X)
        test_loader = DataLoader(TensorDataset(test_X1), batch_size=con.batch_size)

        result = torch.Tensor().to(device)
        model.eval()
        for _data in test_loader:
            data_X = _data[0].to(device)
            # data_X = data_X.unsqueeze(1)
            pred_X = model(data_X)

            result = torch.cat((result, pred_X), dim=0)
        if test_X.shape[0] > num:
            test_X1 = torch.from_numpy(np.reshape(test_X[num:], (test_X.shape[0]-num, 1, con.input_size))).float()
            test_loader = DataLoader(TensorDataset(test_X1), batch_size=test_X.shape[0]-num)
            for _data in test_loader:
                data_X = _data[0].to(device)
                # data_X = data_X.unsqueeze(1)
                pred_X = model(data_X)

                result = torch.cat((result, pred_X), dim=0)

        result = result.detach().cpu().numpy()
        result = Y_scaler.inverse_transform(result)
        result = np.squeeze(result)
        data[:, i+8] = result

    data = pd.DataFrame(data)
    writer = pd.ExcelWriter('result 1.xlsx')
    data.to_excel(writer, index=False)
    writer.save()





