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


# Including configuration, data reading, logging, plotting
class Config:
    # data parameter
    input_columns = [0, 1, 2, 3]
    output_columns = [4]   # Specifies the column name of density

    # network parameter
    input_size = 4
    output_size = 1
    hidden_size = [16, 32, 32]
    dropout_rate = 0.2  # dropout probability. Do not add the dropout layer for now
    batch_first = True  # input dimension order: bacth_size, sequence, input_size

    # Training parameter
    # do_train = True
    do_train = False
    do_test = False #  # Collect new data
    # do_test = True
    add_train = False  # Whether to load existing model parameters for incremental training
    # add_train = True
    pre_model_name = '68-model.pth'  # Load the existing model name
    shuffle_train_data = True  # Whether to shuffle the training data
    # use_cuda = False  # Whether to use GPU training The cuda version must be compatible with the current GPU
    use_cuda = True
    # debug mode
    # debug_mode = True # In debug mode, load a small amount of data
    debug_mode = False

    batch_size = 1000  # Exercise batch_size sets per epoch. The shape of input is ultimately [bacth_size, sequence, input_size].
    train_data_rate = 0.85  # Proportion of training data to total data
    # Validation sets are used during training to make model and parameter selections
    valid_data_rate = 0.15  # Proportion of validation data to training data.  1-train_data_rate

    learning_rate = 0.001
    epoch = 100  # times is the entire training set trained, regardless of the premise of early stop
    patience = 3  # How many epochs are trained, and stop training if the verification set is not improved
    random_seed = 0  # Random seeds, guaranteed reproducible 0 or 42 are more commonly used

    train_data_path = 'D:/seawater/2017_2021/'
    test_data_path = 'D:/seawater/2022/'
    model_name = '-model.pth'  # model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_save_path = './checkpoint/'
    figure_save_path = './figure/'
    log_save_path = './log/'
    do_log_print_to_screen = True
    do_log_save_to_file = True  # Whether to log config and training procedures
    do_figure_save = False
    do_train_visualized = False  # Train loss visualization
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)  # makedirs
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

    # filepath can be a file or a dir
    # dir can be the default path of config or a new path
    def read_data(self, filepath, logger):
        filelist = []
        if os.path.isdir(filepath):
            filelist = [filepath + file for file in os.listdir(filepath) if file.endswith('.xlsx')]
        elif os.path.isfile(filepath) and filepath.endswith('.xlsx'):  # 传入csv file
            filelist = self.config.train_data_path + filepath
        if not filelist:
            return [], []

        file_num = 1 if self.config.debug_mode else len(filelist) # In debug mode, load a small amount of data
        all_data = pd.DataFrame([])
        for file in filelist[0:file_num]:  # Read all the data at once
            # Control data types to reduce memory
            col_type = {'LATITUDE ': np.int16, 'LONGITUDE ': np.int16, 'PRES ': np.float32, 'MONTH':np.int8, 'rho_TEOS10':np.float32}
            data = pd.read_excel(file, dtype=col_type)
            # Filter data up to 2200dbar
            data = data[data['PRES '] <= 2200]
            # data = data[data['PRES '] <= 200]
            # data['rho_TEOS10'] = data['rho_TEOS10'] -1000;
            # logger.info(data.info())
            # logger.info(data.describe())
            if all_data.empty:
                all_data = data
            else:
                all_data = pd.concat([all_data, data], axis=0)
            logger.info(all_data.shape[0])
            # logger.info(all_data.info())

        # # print(all_data.any)
        # # y_sort = np.argsort(all_data[:, 1]) # Sort by latitude
        # # all_data = all_data[y_sort, :]
        # # y variable distribution histogram
        # fig = plt.figure(figsize=(8, 5)) # Set the canvas size
        # plt.rcParams[' font-sans-serif '] = 'SimHei' # Set the Chinese display
        # plt.rcParams['axes.unicode_minus'] = False # Fix save image is negative sign '-' displayed as a box
        # # Drawing histogram bins: Control the number of intervals in the histogram auto to the number of autofill color: Specify the fill color of the column
        # data_tmp = all_data['rho_TEOS10']
        # plt.hist(data_tmp, bins=100, color='g')
        # plt.xlabel('density') # Sets the name of the X-axis
        # plt.ylabel(' quantity ') # Set the Y-axis name
        # plt.title('density') # Set the title name
        # plt.show() # Show pictures
        #
        # fig = plt.figure(figsize=(8, 5)) # Set the canvas size
        # plt.rcParams[' font-sans-serif '] = 'SimHei' # Set the Chinese display
        # plt.rcParams['axes.unicode_minus'] = False # Fix save image is negative sign '-' displayed as a box
        # # Drawing histogram bins: Control the number of intervals in the histogram auto to the number of autofill color: Specify the fill color of the column
        # data_tmp = all_data['LATITUDE ']
        # plt.hist(data_tmp, bins=100, color='g')
        # plt.xlabel('latitude') # Set the X-axis name
        # plt.ylabel(' quantity ') # Set the Y-axis name
        # plt.title('latitude') # Set the title name
        # plt.show() # Show pictures
        #
        # fig = plt.figure(figsize=(8, 5)) # Set the canvas size
        # plt.rcParams[' font-sans-serif '] = 'SimHei' # Set the Chinese display
        # plt.rcParams['axes.unicode_minus'] = False # Fix save image is negative sign '-' displayed as a box
        # # Drawing histogram bins: Control the number of intervals in the histogram auto to the number of autofill color: Specify the fill color of the column
        # data_tmp = all_data['LONGITUDE ']
        # plt.hist(data_tmp, bins=100, color='g')
        # plt.xlabel('longitude') # Set the X-axis name
        # plt.ylabel(' quantity ') # Set the Y-axis name
        # plt.title('longitude') # Set the title name
        # plt.show() # Show pictures
        #
        # fig = plt.figure(figsize=(8, 5)) # Set the canvas size
        # plt.rcParams[' font-sans-serif '] = 'SimHei' # Set the Chinese display
        # plt.rcParams['axes.unicode_minus'] = False # Fix save image is negative sign '-' displayed as a box
        # # Drawing histogram bins: Control the number of intervals in the histogram auto to the number of autofill color: Specify the fill color of the column
        # data_tmp = all_data['PRES ']
        # plt.hist(data_tmp, bins=100, color='g')
        # plt.xlabel('pressure') # Set the X-axis name
        # plt.ylabel(' quantity ') # Set the Y-axis name
        # plt.title('pressure') # Set the title name
        # plt.show() # Show picture

        # Correlation analysis of data
        sns.heatmap(all_data.corr(), cmap="YlGnBu", annot=True)  # heat map
        plt.title('Correlation analysis heat map')
        plt.show()
        return all_data

    def get_train_and_valid_data(self, logger):
        feature_std = self.all_data.values[:, self.config.input_columns]
        output = self.all_data.values[:, self.config.output_columns]
        train_X, valid_X, train_Y, valid_Y = train_test_split(feature_std, output, test_size=1-self.config.train_data_rate, random_state=42,shuffle=self.config.shuffle_train_data)
        # # if add_train == Ture, use the previous scaler
        # X_scaler, Y_scaler = dict_scaler_para()
        # setattr(X_scaler, 'n_samples_seen_', self.data_num)
        # setattr(Y_scaler, 'n_samples_seen_', self.data_num)
        X_scaler = StandardScaler()
        train_X = X_scaler.fit_transform(train_X)
        # train_X = X_scaler.transform(train_X)
        logger.info(X_scaler.mean_)
        logger.info(X_scaler.var_)
        valid_X = X_scaler.transform(valid_X)
        Y_scaler = StandardScaler()
        train_Y = Y_scaler.fit_transform(train_Y)
        # train_Y = Y_scaler.transform(train_Y)
        logger.info(Y_scaler.mean_)
        logger.info(Y_scaler.var_)
        valid_Y = Y_scaler.transform(valid_Y)
        return train_X, valid_X, train_Y, valid_Y

    def get_test_data(self, logger, test_path=None, return_out=False):
        if test_path:
            self.all_data = self.read_data(test_path, logger)
        feature_std = self.all_data.values[:, self.config.input_columns]
        output = self.all_data.values[:, self.config.output_columns]
        if return_out:
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

def draw(config: Config, origin_data: Data, logger, predict_out: np.ndarray):
    origin_out = origin_data.all_data.iloc[origin_data.train_num + origin_data.start_num_in_test:, config.output_columns].values  # 测试集里的out

    assert origin_out.shape[0] == predict_out.shape[0]
    out_name = [i for i in config.out_colums]
    num = len(out_name)

    loss = np.mean((origin_out - predict_out)**2, axis=0)
    loss_norm = loss/np.var(origin_out, axis=0)
    logger.info("The mean squared error of {} is ".format(out_name) + str(loss_norm))

    x = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    if not sys.platform.startswith('linux'):    # Linux without a desktop cannot be output. If it is Linux with a desktop, such as Ubuntu, you can remove this line
        for i in range(num):
            plt.figure(i+1) # Mapping of predictive data
            plt.plot(x, origin_out[:, i], label=out_name[i] + '_origin')
            plt.plot(x, predict_out[:, i], label=out_name[i] + '_predict')
            plt.title('predict for ' + out_name[i])
            plt.legend(['origin', 'predict'])
            logger.info('the predicted {} is:'.format(out_name[i]) + str(np.squeeze(predict_out[:, i])))
            if config.do_figure_save:
                plt.savefig(config.figure_save_path + out_name[i] + 'predict.png')

        plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=config.hidden_size[0], kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)  # maximum pooling layer
        self.conv2 = nn.Conv1d(in_channels=config.hidden_size[0], out_channels=config.hidden_size[1], kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=config.hidden_size[1],out_features=config.hidden_size[2])
        self.fc2 = nn.Linear(in_features=config.hidden_size[2], out_features=config.output_size)
        self.f1 = nn.Sigmoid()
        self.f2 = nn.Tanh()
        self.f3 = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.f1(self.conv1(x)))
        # print(x.size())
        x = self.pool(self.f2(self.conv2(x)))
        # print('******************************')
        # print(x.size())
        x = x.view(-1, 32)  # Dimensional transformation
        x = self.f3(self.fc1(x))
        x = self.fc2(x)
        return x


def train(config, logger, train_and_valid_data):

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    batch_size = config.batch_size
    num_train = np.shape(train_X)[0] - np.shape(train_X)[0] % batch_size
    train_X = torch.from_numpy(np.reshape(train_X[:num_train], (num_train, 1, config.input_size))).float()
    # train_X = torch.from_numpy(np.reshape(train_X[:num_train], (num_train, config.input_size))).float()
    train_Y = torch.from_numpy(np.reshape(train_Y[:num_train], (-1, config.output_size))).float()  # turn to Tensor

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size)

    num_valid = np.shape(valid_X)[0] - np.shape(valid_X)[0] % batch_size
    valid_X = torch.from_numpy(np.reshape(valid_X[:num_valid], (num_valid, 1, config.input_size))).float()
    # valid_X = torch.from_numpy(np.reshape(valid_X[:num_valid], (num_valid, config.input_size))).float()
    valid_Y = torch.from_numpy(np.reshape(valid_Y[:num_valid], (-1, config.output_size))).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=batch_size)
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU or GPU
    # device = torch.device("cpu")
    model = NeuralNetwork(config).to(device)  # For GPU training,.to(device) copies the model/data to GPU memory

    if config.add_train:  # For incremental training, the original model parameters are loaded first
        model.load_state_dict(torch.load(config.model_save_path + config.pre_model_name))
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()  # define the optimizer and loss
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    train_loss = []
    valid_loss = []
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))

        model.train()  # In pytorch, switch to training mode when you train
        train_loss_array = []

        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()  # Set the gradient information to 0 before training
            # print(_train_X.size())
            pred_Y = model(_train_X)  # Computes the forward function forward

            loss = criterion(pred_Y, _train_Y)  # compute loss
            loss.backward()  # Backpropagation of loss
            optimizer.step()  # Optimizer updates parameters
            train_loss_array.append(loss.item())
            global_step += 1
            if global_step % 100 == 0:
                print('Step {}, loss {}'.format(global_step, loss.item()))
            # if config.do_train_visualized and global_step % 100 == 0:   # displayed every 100 steps
            #     vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
            #              update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # The following is the early stop mechanism.
        # When the model training did not improve the prediction effect of the validation set for consecutive periods of
        # config.patience epochs, it was stopped to prevent overfitting.
        model.eval()                    # In pytorch, when forecasting, you switch to prediction mode
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = model(_valid_X)
            loss = criterion(pred_Y, _valid_Y)  # The verification process is only forward calculation without backpropagation
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
            # If the validation set index does not improve for consecutive patience epochs, the training is stopped
            if bad_epoch >= config.patience:
                logger.info(" The training stops early in epoch {}".format(epoch))
                break

    x = range(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, valid_loss)
    plt.legend(['train_loss', 'valid_loss'])
    plt.show()


def predict(config, test_X):
    # Get test data
    X_scaler, Y_scaler = dict_scaler_para()

    num_test = test_X.shape[0] - test_X.shape[0] % config.batch_size
    # use the previous scaler
    test_X = X_scaler.transform(test_X)
    test_X = torch.from_numpy(np.reshape(test_X[:num_test], (num_test, 1, config.input_size))).float()
    test_loader = DataLoader(TensorDataset(test_X), batch_size=config.batch_size)

    # Loading model
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model = NeuralNetwork(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.pre_model_name))  # 加载模型参数

    # define a tensor to hold the prediction
    result = torch.Tensor().to(device)

    model.eval()   # In pytorch, when forecasting, you switch to prediction mode
    for _data in test_loader:
        data_X = _data[0].to(device)
        # data_X = data_X.unsqueeze(1)
        pred_X = model(data_X)
        # if not config.do_continue_train: hidden_predict = None
        # cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, pred_X), dim=0)
    result = result.detach().cpu().numpy()
    result = Y_scaler.inverse_transform(result.reshape(-1, 1))
    return result  # if in the gpu to go to the cpu, and finally return numpy data

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


if __name__ == '__main__':

    # main()
    con = Config()
    logger = load_logger(con)

    try:
        np.random.seed(con.random_seed)  # Set random seeds to ensure reproducibility
        filepath = con.train_data_path  # Easy to specify other paths, file
        data_gainer = Data(con, filepath, logger)
        logger.info('read data from {}'.format(filepath))

        if con.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data(logger)
            train(con, logger, [train_X, train_Y, valid_X, valid_Y])
            # pred_result = predict(con, valid_X)
            # valid_y = np.reshape(valid_Y, [valid_Y.shape[0] * valid_Y.shape[1]])
            # x = range(len(pred_result))
            # plt.plot(x, valid_y, label='origin')
            # plt.plot(x, pred_result, label='_predict')
            # plt.legend(['origin', 'predict'])
            # plt.show()
        if con.do_test:
            # test_X, test_Y = data_gainer.get_test_data(test_path=con.test_data_path,return_out=True)
            test_X, test_Y = data_gainer.get_test_data(logger=logger, return_out=True)
            pred_result = predict(con, test_X)
            error = pred_result-test_Y
            logger.info('ME: {}'.format(np.mean(error)))
            logger.info('MAE: {}'.format(np.mean(np.abs(error))))
            logger.info('RMSE: {}'.format(np.std(error)))

            x = range(len(pred_result))
            plt.plot(x, test_Y, label='origin')
            plt.plot(x, pred_result, label='predict')
            plt.legend(['origin', 'predict'])
            plt.show()
            plt.scatter(test_X[:, 0], error, label='error')
            plt.show()
            draw(con, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)
