import numpy as np

class Uniform_Agent():
    def __init(self,n_stocks):
        self.n_stocks = n_stocks

    def act(self):

        Ones = np.ones(self.n_stocks+1)

        return Ones/Ones.sum()