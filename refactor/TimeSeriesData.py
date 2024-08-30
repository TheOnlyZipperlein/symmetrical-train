import numpy as np

class TimeSeriesData:
    def __init__(self, data:np.ndarray, split:tuple, lookback:int): 
        self.lookback_window = lookback
        self.data = data[lookback:]
        self.split = split
        self.ann_X = list()
        self.ann_y = list()
        for i in range(len(data)): 
            if(i>=lookback):
                self.ann_X.append(data[i-lookback:i])
                self.ann_y.append(i)
        self.ann_X = np.array(self.ann_X)
        self.ann_y = np.array(self.ann_y)


    def get_train(self, context:str):
        train_size = len(self.data) - int(len(self.data)*0.5)
        match context:
            case "ARIMA":                
                return self.data[:train_size]
            case "ANN":
                return self.ann_X[:train_size], self.ann_y[:train_size]
            case _:
                raise ValueError("")

    def get_val(self, context:str):        
        train_size = len(self.data) - int(len(self.data)*0.5)
        val_size = len(self.data) - int(len(self.data)*0.25)
        match context:
            case "ARIMA":
                return self.data[train_size:train_size+val_size]
            case "ANN":
                return self.ann_X[train_size:train_size+val_size], self.ann_y[train_size:train_size+val_size]
            case _:
                raise ValueError("")

    def get_test(self, context:str):
        train_size = len(self.data) - int(len(self.data)*0.5)
        val_size = len(self.data) - int(len(self.data)*0.25)
        match context:
            case "ARIMA":                
                return self.data[train_size+val_size:]
            case "ANN":
                return self.ann_X[train_size+val_size:], self.ann_y[train_size+val_size:]
            case _:
                raise ValueError("")
    
    def get_all(self, context:str):
        match context:
            case "ARIMA":                
                return self.data
            case "ANN":
                return self.ann_X, self.ann_y
            case _:
                raise ValueError("")

######### Debug Code ###############
import pandas as pd

df = pd.read_csv(r".\data\airline-passengers.csv")
ts = df[["Passengers"]].values.astype("float32")

ts = TimeSeriesData(ts, [0.72, 0.14, 0.14], 14)
print("")