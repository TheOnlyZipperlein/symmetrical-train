from sklearn.metrics import mean_absolute_error

import numpy as np
import sys
import warnings
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from TimeSeriesData import TimeSeriesData
from GANParameterOptimizer import GANParameterOptimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IWrapper import IWrapper

class FlexANN(nn.Module):
    def __init__(self, lookback_window, seed, bias, w_low, w_high, depth, hidden_size, parallel_dimensions_count):
        super(FlexANN, self).__init__()
        self.seed = seed
        self.weight_limit = (w_low, w_high)
        self.bias_init = bias
        self.fc_dimensions = nn.ModuleList()
        
        for h in range(parallel_dimensions_count): 
            dimension =  nn.ModuleList()
            dimension.append(nn.Linear(lookback_window, hidden_size, bias=True))
            for i in range(depth):
                dimension.append(nn.Linear(hidden_size, hidden_size, bias=True))
            self.fc_dimensions.append(dimension)
        self.end = nn.Linear(hidden_size*parallel_dimensions_count, 1, bias=True)

        self.init_bias()
    def forward(self, x:torch.Tensor):
        x = x.squeeze()
        x_l = []
        for d in self.fc_dimensions:
            x_t = x
            for fc in d:
                x_t = F.relu(fc(x_t))
            x_l.append(x_t)
        x = torch.cat(x_l, dim=1)
        x = self.end(x)
        return x
    
    def init_weights(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        w_low, w_high = self.weight_limit

        for layer in self.fc_layers:
            if layer.bias is not None:
                nn.init.uniform_(layer.weight, a=-w_low, b=w_high)

        if self.end.bias is not None:
            nn.init.constant_(self.end.bias, self.bias_init)
    
    def init_bias(self):
        for d in self.fc_dimensions:
            for layer in d:
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, self.bias_init)

        if self.end.bias is not None:
            nn.init.constant_(self.end.bias, self.bias_init)
    
class ANN_Wrapper(IWrapper):
    def __init__(self, data:TimeSeriesData, functions_to_keep:int):
        self.functions_to_keep = functions_to_keep
        self.data = data

    def grid_search_best_functions(self, params:tuple[int, int]):
        raise ValueError("Probbably not better")

    def random_search_best_functions(self, limits:tuple[int, int, int], sample_size:int):
        seed_lower_limit, seed_upper_limit = (0, 200)
        bias_lower_limit, bias_upper_limit = (0.2, 0.8)
        w_low_lower_limit, w_low_upper_limit = (0.2, 0.49)
        w_high_lower_limit, w_high_upper_limit = (0.5, 0.8)
        depth_lower_limit, depth_upper_limit = (2, 5)
        hidden_size_lower_limit, hidden_size_upper_limit = (20, 1000)        
        parallel_dimensions_count_lower_limit, parallel_dimensions_count_upper_limit = (2, 10)
        learning_rate_lower_limt, learning_rate_upper_limt = (0.01, 0.05)
        epochs_lower_limit, epochs_upper_limit = (40, 200)
        batch_size_lower_limit, batch_size_upper_limit = (5, 20)
        
        X_train, y_train = self.data.get_train(context = "ANN")
        X_train, y_train =  torch.tensor(np.array(X_train).astype(np.float32)).cuda(), torch.tensor(np.array(y_train).astype(np.float32)).cuda()
        X_val, y_val = self.data.get_val(context = "ANN")
        X_val = torch.tensor(np.array(X_val).astype(np.float32)).cuda()

        models = list()
        scores = list()
        criterion = nn.MSELoss(reduction='mean')

        start_time = time.time()
        c = 0
        
        for i in range(sample_size):
            seed = random.randint(seed_lower_limit, seed_upper_limit)
            bias = random.uniform(bias_lower_limit, bias_upper_limit)
            w_low = random.uniform(w_low_lower_limit, w_low_upper_limit)
            w_high = random.uniform(w_high_lower_limit, w_high_upper_limit)
            depth = random.randint(depth_lower_limit, depth_upper_limit)
            hidden_size = random.randint(hidden_size_lower_limit, hidden_size_upper_limit)
            parallel = random.randint(parallel_dimensions_count_lower_limit, parallel_dimensions_count_upper_limit)

            model = FlexANN(self.data.lookback_window, seed, bias, w_low, w_high, depth, hidden_size, parallel)
            model = model.cuda()

            batch_size = random.randint(batch_size_lower_limit, batch_size_upper_limit)
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=0.0003)

            model.train()
            epochs = random.randint(epochs_lower_limit, epochs_upper_limit)
            learning_rate = random.uniform(learning_rate_lower_limt, learning_rate_upper_limt)
            for i in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad:
                y_pred = model(X_val)            
            mea = mean_absolute_error(y_val, y_pred.numpy())

            params = {
                "seed": seed,
                "bias": bias,
                "w_low": w_low,
                "w_high": w_high,
                "depth": depth,
                "hidden_size": hidden_size,
                "parallel": parallel,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate
            }

            model.parameters_a = params
            model.score_ = mea
            scores.append(mea)
            models.append(model)
                        
            c=c+1
            current_time = time.time()
            if current_time - start_time >= 2:
                print(f"Comparing {sample_size} ANN Models. Done: {c/sample_size:.2%}", end='\r')
                start_time = current_time
        
        # Order models by scores
        models = [model for _, model in sorted(zip(scores, models), key=lambda pair: pair[0])]

        self.best_models = models[:self.functions_to_keep]

    def gan_improve_functions(self, epochs:int):
        gan = GANParameterOptimizer(3)
        ones = [1] * len(self.best_models)
        parameter = [model.parameters_a for model in self.best_models]
        gan.train(parameter, ones, epochs, self)

        # Order models by scores
        arima_models = [model for model in sorted(self.best_models, key=lambda pair: pair.score_)]

        self.best_models = arima_models[:self.functions_to_keep]

    def evaluate(self, parameter_list):
        pass

    def drop_models():
        pass

    def get_predictions(self, context:str):
        match context:
            case "train":
                pass
            case _:
                raise ValueError("")
            
######### Debug Code ###############
import pandas as pd

df = pd.read_csv(r".\data\airline-passengers.csv")
ts = df[["Passengers"]].values.astype("float32")

wrapper = ANN_Wrapper(TimeSeriesData(ts, [0.72, 0.14, 0.14], 14), 10)
wrapper.random_search_best_functions((100,100,1), 2)
#wrapper.gan_improve_functions(2, 5)
print("test")