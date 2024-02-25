import numpy as np
from .strategy import Strategy

class CyclicOutputDependecy(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(CyclicOutputDependecy, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        uncertainty = self.predict_uncertainty(unlabeled_data).to('cpu')
        return np.argsort(uncertainty)[-n:]
