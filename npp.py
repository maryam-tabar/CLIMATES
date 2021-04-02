import numpy as np
import scipy.signal
import theano.tensor as T
from cle.cle.data import TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import totuple
from build import load_data
#### The code is adapted from https://github.com/jych/nips2015_vrnn which is linked to the following reference.
##Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A.C., Bengio, Y.: A recurrent latent variable model for sequential data. Advances in neural information processing systems 28, 2980–2988 (2015)
##

class NPP(TemporalSeries, SequentialPrepMixin):
    """
    NPP dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, prep='none', X_mean=None, X_std=None,
                 bias=None, **kwargs):

        self.prep = prep
        self.X_mean = X_mean
        self.X_std = X_std
        self.bias = bias

        super(NPP, self).__init__(**kwargs)

    def load(self, data_path):

        if self.name == "train":
            X, Y, _, _, _, _, _, mean_data, std_data = load_data(data_path)
            print("train")
        elif self.name == "valid":
            _, _, X, Y, _, _, _, mean_data, std_data = load_data(data_path)
            print("valid")

        return [X, Y]

    def theano_vars(self):
        return [T.ftensor3('x'), T.ftensor3('y')]

    def slices(self, start, end):
        batches = [mat[start:end].swapaxes(0, 1) for mat in self.data]
        return totuple([batches[0], batches[1]])

    def generate_index(self, X):

        maxlen = np.array([len(x) for x in X]).max()
        idx = np.arange(maxlen)

        return idx
        
