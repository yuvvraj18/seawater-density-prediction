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


# 包括配置，数据读取，日志记录，绘图
class Config:
    # 数据参数
    input_columns = [0, 1, 2, 3]
    output_columns = [4]   # data中BP所在列名

    # 网络参数
    input_size = 4
    output_size = 1
    hidden_size = [16, 32, 32]
    dropout_rate = 0.2  # dropout概率 暂时不添加dropout层
    batch_first = True  # input维度顺序  (bacth_size, sequence, input_size)


    # 训练参数
    # do_train = True
    do_train = False
    # do_test = False #  # 采集新的数据进行BP测量
    do_test = True
    add_train = False  # 是否载入已有模型参数进行增量训练
    # add_train = True
    pre_model_name = '68-model.pth'  # 加载已有模型名字
    shuffle_train_data = True  # 是否对训练数据做shuffle
    # use_cuda = False  # 是否使用GPU训练 cuda版本与当前GPU不兼容
    use_cuda = True

    # 训练模式
    # debug_mode = True
    debug_mode = False  # 调试模式下，一些draw print在debug模式下进行

    batch_size = 500  # 每一次epoch训练30组  input的shape最终为 [bacth_size, sequence, input_size)
    train_data_rate = 0.85  # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.2  # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择
    learning_rate = 0.001
    epoch = 100  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 3  # 训练多少epoch，验证集没提升就停掉
    random_seed = 42  # 随机种子，保证可复现  0或42比较常用

    train_data_path = 'D:/MatlabWork/seawater/2022/'
    # train_data_path = 'D:/Queensland data/'
    test_data_path = 'D:/MatlabWork/seawater/2022e/'
    model_name = '-model.pth'  # model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_save_path = './checkpoint/'
    figure_save_path = './figure/'
    log_save_path = './log/'
    do_log_print_to_screen = True
    do_log_save_to_file = True  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False  # 训练loss可视化，后面再学习tensorBoard用法
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)  # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + " "
        os.makedirs(log_save_path)

class Data:
    def __init__(self, config, filepath, logger):
        self.config = config
        self.all_data = self.read_data(filepath, logger)
        self.data_column_name = self.all_data.columns.to_list()   # .columns.tolist() 是获取列名

        self.data_num = self.all_data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

    # filepath可以是一个文件 或者一个dir
    # dir 可以是config的默认路径 也可以是新的路径
    def read_data(self, filepath, logger):
        filelist = []
        if os.path.isdir(filepath):  # 传入dir
            filelist = [filepath + file for file in os.listdir(filepath) if file.endswith('.xlsx')]
        elif os.path.isfile(filepath) and filepath.endswith('.xlsx'):  # 传入csv file
            filelist = self.config.train_data_path + filepath
        if not filelist:
            return [], []

        file_num = 2 if self.config.debug_mode else len(filelist)
        # all_data = pd.read_excel(filelist[file_num])
        # all_data = all_data[(all_data['PRES '] <= 2200)]
        all_data = pd.DataFrame([])
        for file in filelist[:file_num]:  # 一次读入所有数据 [1:file_num]
            data = pd.read_excel(file)
            # 数据筛选 按照纬度、经度、压力排序 得到每个位置的起点和终点  判断深度范围是否在1000m以上
            data = data[(data['PRES '] <= 2200)]
            # logger.info(data.info())
            # logger.info(data.describe())
            if all_data.empty:
                all_data = data
            else:
                all_data = pd.concat([all_data, data], axis=0)
            logger.info(all_data.shape[0])

        # print(all_data.any)
        # y_sort = np.argsort(all_data[:, 1]) # 按照纬度排序
        # all_data = all_data[y_sort, :]
        # y变量分布直方图

        # # 数据的相关性分析
        # sns.heatmap(all_data.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
        # plt.title('相关性分析热力图')  # 设置标题名称
        # plt.show()  # 展示图片

        return all_data

    def get_train_and_valid_data(self, logger):
        X_scaler, Y_scaler = dict_scaler_para()
        setattr(X_scaler, 'n_samples_seen_', self.data_num)
        setattr(Y_scaler, 'n_samples_seen_', self.data_num)
        data = pd.DataFrame(X_scaler.fit_transform(self.all_data.values))
        # data = self.all_data
        feature_std = data.values[:, self.config.input_columns]
        output = data.values[:, self.config.output_columns]
        train_X, valid_X, train_Y, valid_Y = train_test_split(feature_std, output, test_size=1-self.config.train_data_rate, random_state=42,shuffle=self.config.shuffle_train_data)
        return train_X, valid_X, train_Y, valid_Y

    def get_test_data(self, logger, test_path=None, return_BP=False):
        if test_path:
            self.all_data = self.read_data(test_path, logger)
        feature_std = self.all_data.values[:, self.config.input_columns]
        # feature_std = (feature_std + np.array([-0.5, -0.5, 0, 0]))*np.array([180, 360, 1e4, 12])
        output = self.all_data.values[:, self.config.output_columns]
        # output = output * 100 + 1000
        if return_BP:  # 实际应用中的测试集是没有label数据的
            return np.array(feature_std), output
        return np.array(feature_std)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    config_dict = {}
    for key in dir(config):
        if not key.startswith("_"):
            config_dict[key] = getattr(config, key)
    config_str = str(config_dict)
    config_list = config_str[1:-1].split(", '")
    config_save_str = "\nConfig:\n" + "\n'".join(config_list)
    logger.info(config_save_str)

    return logger

def draw(config: Config, origin_data: Data, logger, predict_BP: np.ndarray):
    origin_BP = origin_data.all_data.iloc[origin_data.train_num + origin_data.start_num_in_test:, config.output_columns].values  # 测试集里的BP

    assert origin_BP.shape[0] == predict_BP.shape[0]
    BP_name = [i for i in config.BP_colums]
    num = len(BP_name)

    loss = np.mean((origin_BP - predict_BP)**2, axis=0)
    loss_norm = loss/np.var(origin_BP, axis=0)
    logger.info("The mean squared error of {} is ".format(BP_name) + str(loss_norm))

    x = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(num):
            plt.figure(i+1) # 预测数据绘制
            plt.plot(x, origin_BP[:, i], label=BP_name[i] + '_origin')
            plt.plot(x, predict_BP[:, i], label=BP_name[i] + '_predict')
            plt.title('predict for ' + BP_name[i])
            plt.legend(['origin', 'predict'])
            logger.info('the predicted {} is:'.format(BP_name[i]) + str(np.squeeze(predict_BP[:, i])))
            if config.do_figure_save:
                plt.savefig(config.figure_save_path + BP_name[i] + 'predict.png')

        plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=config.hidden_size[0], kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)  # 一维最大池化层
        self.conv2 = nn.Conv1d(in_channels=config.hidden_size[0], out_channels=config.hidden_size[1], kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=config.hidden_size[1],out_features=config.hidden_size[2])
        self.fc2 = nn.Linear(in_features=config.hidden_size[2], out_features=config.output_size)
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Tanh()
        self.f3 = nn.ReLU()

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.f1(x)
        # x = self.pool(x)
        x = self.pool(self.f1(self.conv1(x)))
        # print(x.size())
        # x = self.conv2(x)
        # x = self.f2(x)
        # x = self.pool(x)
        x = self.pool(self.f2(self.conv2(x)))
        # print('******************************')
        # print(x.size())
        x = x.view(-1, 32)  # 维度变换
        # x = self.fc1(x)
        # x = self.f3(x)
        x = self.f3(self.fc1(x))
        x = self.fc2(x)
        return x

def train(config, logger, train_and_valid_data):

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    batch_size = config.batch_size
    num_train = np.shape(train_X)[0] - np.shape(train_X)[0] % batch_size
    train_X = torch.from_numpy(np.reshape(train_X[:num_train], (num_train, 1, config.input_size))).float()
    train_Y = torch.from_numpy(np.reshape(train_Y[:num_train], (-1, config.output_size))).float()  # 先转为Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size)

    num_valid = np.shape(valid_X)[0] - np.shape(valid_X)[0] % batch_size
    valid_X = torch.from_numpy(np.reshape(valid_X[:num_valid], (num_valid, 1, config.input_size))).float()
    valid_Y = torch.from_numpy(np.reshape(valid_Y[:num_valid], (-1, config.output_size))).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=batch_size)
    # device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
    device = torch.device("cpu")
    model = NeuralNetwork(config).to(device)  # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中

    if config.add_train:  # 如果是增量训练，会先加载原模型参数
        model.load_state_dict(torch.load(config.model_save_path + config.pre_model_name))
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()  # 这两句是定义优化器和loss
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    train_loss = []
    valid_loss = []
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))

        model.train()  # pytorch中，训练时要转换成训练模式
        train_loss_array = []

        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()  # 训练前要将梯度信息置 0
            # print(_train_X.size())
            pred_Y = model(_train_X)  # 这里走的就是前向计算forward函数

            loss = criterion(pred_Y, _train_Y)  # 计算loss
            loss.backward()  # 将loss反向传播
            optimizer.step()  # 用优化器更新参数
            train_loss_array.append(loss.item())
            global_step += 1
            if global_step % 100 == 0:
                print('Step {}, loss {}'.format(global_step, loss.item()))
            # if config.do_train_visualized and global_step % 100 == 0:   # 每一百步显示一次
            #     vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
            #              update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = model(_valid_X)
            loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                    "The valid loss is {:.6f}.".format(valid_loss_cur))
        train_loss.append(train_loss_cur)
        valid_loss.append(valid_loss_cur)

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + str(epoch) + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                logger.info(" The training stops early in epoch {}".format(epoch))
                break

    x = range(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, valid_loss)
    plt.legend(['train_loss', 'valid_loss'])
    plt.show()


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
    test_X = torch.from_numpy(np.reshape(test_X[:num_test], (num_test, 1, config.input_size))).float()
    test_loader = DataLoader(TensorDataset(test_X), batch_size=config.batch_size)

    # 加载模型
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model = NeuralNetwork(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.pre_model_name))  # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()   # pytorch中，预测时要转换成预测模式
    for _data in test_loader:
        data_X = _data[0].to(device)
        # data_X = data_X.unsqueeze(1)
        pred_X = model(data_X)
        # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
        # cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, pred_X), dim=0)
    result = result.detach().cpu().numpy()
    result = Y_scaler.inverse_transform(result.reshape(-1, 1))
    return result  # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据


if __name__ == '__main__':
    con = Config()
    logger = load_logger(con)

    try:
        np.random.seed(con.random_seed)  # Set random seeds to ensure reproducibility
        filepath = con.train_data_path
        # data_gainer = Data(con, filepath, logger)
        logger.info('read data from {}'.format(filepath))

        # if con.do_train:
        #     train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data(logger)
        #     train(con, logger, [train_X, train_Y, valid_X, valid_Y])

        if con.do_test:
            filelist = [filepath + file for file in os.listdir(filepath) if file.endswith('.xlsx')]
            file_num = 1 if con.debug_mode else len(filelist)
            for i in range(31, file_num):
                col_type = {'LATITUDE ': np.int16, 'LONGITUDE ': np.int16, 'PRES ': np.float32, 'MONTH': np.int8,
                            'rho_TEOS10': np.float32}
                data = pd.read_excel(filelist[i], dtype=col_type)
                data = data[data['PRES '] <= 2200].values
                logger.info(data.shape[0])
                num_test = data.shape[0] - data.shape[0] % con.batch_size
                pred_result = predict(con, data[:num_test, :4])
                error = pred_result - np.reshape(data[:num_test, 4], (-1, 1))
                logger.info('ME: {}'.format(np.mean(error)))
                logger.info('MAE: {}'.format(np.mean(np.abs(error))))
                logger.info('RMSE: {}'.format(np.std(error)))
                error = np.concatenate([data[:num_test, :], error], axis=1)
                e = pd.DataFrame(error, columns=['LATITUDE ', 'LONGITUDE ', 'PRES ', 'MONTH', 'rho_TEOS10', 'error'])
                e['LATITUDE '] = e['LATITUDE '].astype(np.int16)
                e['LONGITUDE '] = e['LONGITUDE '].astype(np.int16)
                e['MONTH'] = e['MONTH'].astype(np.int16)
                writer = pd.ExcelWriter(con.test_data_path + 'error_' + str(i) +'.xlsx')
                e.to_excel(writer, index=False)
                writer.save()

    except Exception:
        logger.error("Run Error", exc_info=True)

