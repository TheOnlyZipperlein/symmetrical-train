import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import curve_fit
import math

# Load data
df = pd.read_csv(r".\data\airline-passengers.csv")
timeseries = df[["Passengers"]].values.astype("float32")

class LSTMModel(nn.Module):
    def __init__(self, hidden_size, seed_hn, seed_cn, lookback):
        super(LSTMModel, self).__init__()
        self.seed_hn = seed_hn
        self.seed_cn = seed_cn
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size * lookback, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        batch_size = x.size(0)
        torch.manual_seed(self.seed_hn)
        h_0 = torch.randn(size = (1, batch_size, self.hidden_size))
        torch.manual_seed(self.seed_cn)
        c_0 = torch.randn(size = (1, batch_size, self.hidden_size))
        x, (hn, cn) = self.lstm(x, (h_0, c_0))
        x = x.reshape(batch_size, -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x, hn, cn

class ANNModel(nn.Module):
    def __init__(self, hidden_size):
        super(ANNModel, self).__init__()
        self.linear1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.linear2 = nn.Linear(int(hidden_size / 2), 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        return x

class RESmodel(nn.Module):
    def __init__(self):
        super(RESmodel, self).__init__()
        self.linear1 = nn.Linear(5, 30)
        self.linear2 = nn.Linear(30, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        return x
    
class PredModel(nn.Module):
    def __init__(self, inputsize):
        self.inputsize = inputsize
        super(PredModel, self).__init__()
        self.linear1 = nn.Linear(inputsize, inputsize*3)
        self.linear2 = nn.Linear(inputsize*3, inputsize)
        self.linear3 = nn.Linear(inputsize, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        return x

def train_lstm(model, criterion, optimizer, train_loader):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred, _, _ = model(X_batch.float())
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    return model

def train_ann(model, criterion, optimizer, state, y):
    model.train()
    optimizer.zero_grad()
    y_pred = model(state)
    loss = criterion(y_pred.squeeze(), y)
    loss.backward()
    optimizer.step()
    return model

def train_res(pred_model, hn_ver_model, cn_ver_model, res_model, pred_ann_model, y_arima_train, criterion, optimizer, train_loader):
    res_model.train()
    for i, data in enumerate(train_loader):
        X_batch, y_batch = data
        with torch.no_grad():
            pred, hn, cn = pred_model(X_batch)
            h_pred = hn_ver_model(hn)
            c_pred = cn_ver_model(cn)
            pa_pred = pred_ann_model(X_batch)
            arima_pred =  y_arima_train[i*train_loader.batch_size:i*train_loader.batch_size+len(y_batch)]
            X_joined = torch.stack([pred.view(pred.size(0)), h_pred.view(h_pred.size(1)), c_pred.view(c_pred.size(1)), pa_pred.view(pred.size(0)), arima_pred.view(arima_pred.size(0))], 1)
        optimizer.zero_grad()
        y_pred = res_model(X_joined)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    return res_model

def train_ann_pred(model, criterion, optimizer, train_loader):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    return model

def predict_val(pred_model, hn_ver_model, cn_ver_model, res_model, pred_ann_model, y_arima_val, X):
    with torch.no_grad():
        pred, hn, cn = pred_model(X)
        h_pred = hn_ver_model(hn)
        c_pred = cn_ver_model(cn)
        pa_pred = pred_ann_model(X)
        arima_pred =  y_arima_val
        X_joined = torch.stack([pred.view(pred.size(0)), h_pred.view(h_pred.size(1)), c_pred.view(c_pred.size(1)), pa_pred.view(pred.size(0)), arima_pred.view(arima_pred.size(0))], 1)
        return res_model(X_joined)
    
def get_arima_train(train, p, d, q, lookback):
    arima_model = ARIMA(train, order=(p,d,q))
    arima_model = arima_model.fit()
    re = arima_model.predict(start=lookback + 1, end=len(train))   
    return torch.tensor(np.array(re).astype(np.float32)).squeeze()

def get_arima_val(train, val, p, d, q, lookback):
    arima_model = ARIMA(train, order=(p,d,q))
    arima_model = arima_model.fit()    
    start_val = len(train) + lookback + 1
    re =  arima_model.predict(start=start_val, end = len(train) + len(val))
    return torch.tensor(np.array(re).astype(np.float32)).squeeze()

def get_arima_test(train, val, test, p, d, q, lookback):
    arima_model = ARIMA(train, order=(p,d,q))
    arima_model = arima_model.fit()
    start_test = len(train) + len(val) + lookback + 1
    re =  arima_model.predict(start=start_test, end= len(train) + len(val) + len(test))
    return torch.tensor(np.array(re).astype(np.float32)).squeeze()

def train_run(params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Extract hyperparameter
    hidden_size = 10
    seed_cn = int(params["seed_cn"])
    seed_hn = int(params["seed_hn"])
    lookback = int(params["lookback"])
    train_epochs = int(params["train_epochs"])
    batch_size = int(params["batch_size"])
    arima_p = int(params["arima_p"])
    arima_d = int(params["arima_d"])
    arima_q = int(params["arima_q"])
    #depth_ann = int(params["depth_ann"])
    #depth_res = int(params["depth_res"])

    # Train-test split
    train_size = int(len(timeseries) * 0.8)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    train_size = int(len(train) * 0.8)
    val_size = len(train) - train_size
    train, val = train[:train_size], train[train_size:]

    def create_dataset(dataset, lookback):
        X, y = [], []
        for i in range(len(dataset)-lookback):
            features = dataset[i:i+lookback]
            features = np.array([np.float32(array[0]) for array in features])            
            target = dataset[i+lookback:i+lookback+1][0][0]
            X.append(features)
            y.append(target)      
        X = torch.tensor(np.array(X).astype(np.float32))
        y = torch.tensor(np.array(y).astype(np.float32))
        return X, y 
    
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_val, y_val = create_dataset(val, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    pred_lstm_model = LSTMModel(hidden_size, seed_hn, seed_cn, lookback).to(device)
    hn_ver_model = ANNModel(hidden_size).to(device)
    cn_ver_model = ANNModel(hidden_size).to(device)
    ver_model = RESmodel().to(device)
    pred_ann_model = PredModel(lookback).to(device)
    #ARIMA
    y_arima_train= get_arima_train(train, arima_p, arima_d, arima_q, lookback)
    y_arima_val = get_arima_val(train, val, arima_p, arima_d, arima_q, lookback)
    y_arima_test= get_arima_test(train,val, test, arima_p, arima_d, arima_q, lookback)

    #NN
    for _ in range(train_epochs):
        criterion = nn.MSELoss(reduction='mean')
        
        optimizer_lstm = torch.optim.Adam(pred_lstm_model.parameters(), lr=0.0015, weight_decay=0.0003)
        for _ in range(train_epochs):
            pred_lstm_model = train_lstm(pred_lstm_model, criterion, optimizer_lstm, train_loader)

        optimizer_hn = torch.optim.Adam(hn_ver_model.parameters(), lr=0.0015, weight_decay=0.0003)
        for _ in range(train_epochs):
            for X_batch, y_batch in train_loader:
                _, hn, _ = pred_lstm_model(X_batch)
                hn_ver_model = train_ann(hn_ver_model, criterion, optimizer_hn, hn, y_batch)

        optimizer_cn = torch.optim.Adam(cn_ver_model.parameters(), lr=0.0015, weight_decay=0.0003)
        for _ in range(train_epochs):
            for X_batch, y_batch in train_loader:
                _, _, cn = pred_lstm_model(X_batch)
                cn_ver_model = train_ann(cn_ver_model, criterion, optimizer_cn, cn, y_batch)

        optimizer_pred_ann = torch.optim.Adam(pred_ann_model.parameters(), lr=0.0015, weight_decay=0.0003)
        for _ in range(train_epochs):
            pred_ann_model = train_ann_pred(pred_ann_model, criterion, optimizer_pred_ann, train_loader)

        optimizer_ver = torch.optim.Adam(ver_model.parameters(), lr=0.0015, weight_decay=0.0003)
        for _ in range(train_epochs):
            ver_model = train_res(pred_lstm_model, hn_ver_model, cn_ver_model, ver_model, pred_ann_model, y_arima_train, criterion, optimizer_ver, train_loader)

        pred_lstm_model.eval()
        hn_ver_model.eval()
        cn_ver_model.eval()
        ver_model.eval()
        pred_ann_model.eval()        

        with torch.no_grad():
            # Validate time series
            y_pred = predict_val(pred_lstm_model, hn_ver_model, cn_ver_model, ver_model, pred_ann_model, y_arima_val, X_val)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            max_error = torch.max(torch.abs(y_val - y_pred))            
            mse = mean_squared_error(y_val, y_pred)    
    return {
        "loss": -mae,
        "status": STATUS_OK,
        # -- store other results like this
        "scores": (mae, mse, r2, max_error),
        "models": (pred_lstm_model, hn_ver_model, cn_ver_model, ver_model, pred_ann_model),
        "X": (X_train, X_val, X_test),
        "y": (y_train, y_val, y_test),
        "arima": (y_arima_train, y_arima_val, y_arima_test)
    }

hyperopt_loops = 3

# Define the hyperparameter space
param_space = {
    #"hidden_size" : hp.randint("hidden_size", 1, 2),    <--- error?
    "seed_cn" : hp.randint("seed_cn", 1, 100),
    "seed_hn" : hp.randint("seed_hn", 1, 100),
    "lookback" : hp.randint("lookback", 7, 14),
    "train_epochs" : hp.randint("train_epochs", 5, 10),
    "batch_size" : hp.randint("batch_size", 6, 10),
    "arima_p" : hp.randint("arima_p", 0, 4),
    "arima_d" : hp.randint("arima_d", 0, 5),
    "arima_q" : hp.randint("arima_q", 0, 5),
    #"depth_ann" : hp.randint("depth_ann", 2, 4),
    #"depth_res" : hp.randint("depth_res", 2, 4),   
}


trials = Trials()
fmin(
    train_run,
    param_space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=hyperopt_loops)
    
#Find best
scores = list()
models = list()
best_trial = trials.trials[0]
best_mae, _, _, _ = trials.trials[0]["result"]["scores"]
for trial in trials.trials: 
    score = trial["result"]["scores"]
    mae, mse, r2, max_error = score
    if(mae < best_mae):
        best_trial = trial
        best_mae, _, _, _ = score

pred_lstm_model, hn_ver_model, cn_ver_model, ver_model, pred_ann_model = best_trial["result"]["models"]
X_train, X_val, X_test = best_trial["result"]["X"]
y_train, y_val, y_test = best_trial["result"]["y"]
y_arima_train, y_arima_val, y_arima_test  = best_trial["result"]["arima"]

with torch.no_grad():
    y_pred = predict_val(pred_lstm_model, hn_ver_model, cn_ver_model, ver_model, pred_ann_model, y_arima_test, X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    max_error = torch.max(torch.abs(y_test - y_pred))            
    mse = mean_squared_error(y_test, y_pred)
print(mae)
y_pred_test = y_pred
y_pred_train = y_pred = predict_val(pred_lstm_model, hn_ver_model, cn_ver_model, ver_model, pred_ann_model, y_arima_val, X_val)

print("")
