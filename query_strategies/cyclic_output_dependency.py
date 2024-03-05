import numpy as np
from .strategy import Strategy

class CyclicOutputDependecy(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(CyclicOutputDependecy, self).__init__(dataset, net, args_input, args_task)

    def query(self, n, rd):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        uncertainty = self.predict_uncertainty(unlabeled_data).to('cuda')
        uncertainty = uncertainty.cpu()

        if rd == 1:
            query_idxs = np.argsort(uncertainty.numpy())[:n]
        
        else:
            query_idxs = np.argsort(uncertainty.numpy())[-n:]


        return unlabeled_idxs[query_idxs]
