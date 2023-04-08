import os
import pandas as pd
import numpy as np

class Dataloader:

    def __init__(self) -> None:
        self.s = 'data/dataset/iris.data'

    def read_data(self):

        df = pd.read_csv(self.s,
                 header=None,
                 encoding='utf-8')
        
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', 0, 1)

        X = df.iloc[0:100, [0,2]].values
        
        return X, y
    
    def standardization(self, X):
        X_std = np.copy(X)
        X_std[:,0] = (X[:,0] - X[:,0].mean())/ X[:,0].std()
        X_std[:,1] = (X[:,1] - X[:,1].mean())/ X[:,1].std()

        return X_std

