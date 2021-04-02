#### The code is adapted from https://github.com/jych/nips2015_vrnn which is linked to the following reference.
##Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A.C., Bengio, Y.: A recurrent latent variable model for sequential data. Advances in neural information processing systems 28, 2980–2988 (2015)
##

import os
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
import numpy as np
import theano
import theano.tensor as T
from cle.cle.cost import MSE, Gaussian, KLGaussianGaussian
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping,
    WeightNorm
)
from cle.cle.train.opt import RMSProp, Adam
from cle.cle.utils import init_tparams, sharedX
from cle.cle.utils.compat import OrderedDict
from cle.cle.utils.op import Gaussian_sample
from cle.cle.utils.gpu_op import concatenate

from datasets.npp import NPP


def main():
    for rnn_dim in [256]:
        print(rnn_dim)
        pkl_name = 'LSTM_hidden{}'.format(rnn_dim)
        npp_path = './datasets/npp.npy'
        save_path = './models/'
        force_saving_freq = 1
        epoch = 1
        batch_size = 256
        x_dim = 1
        z_dim = 64
        lr = 0.001

        p_x_dim = 64
        x2s_dim = 64
        z2s_dim = 64
        q_z_dim = 64
        p_z_dim = 64
        target_dim = x_dim

        model = Model()

        init_W = InitCell('rand')
        init_U = InitCell('ortho')
        init_b = InitCell('zeros')
        init_b_sig = InitCell('const', mean=0.6)

        train_data = NPP(name='train', path=npp_path)

        x, y = train_data.theano_vars()

        x_1 = FullyConnectedLayer(name='x_1',
                                  parent=['x_t'],
                                  parent_dim=[x_dim],
                                  nout=x2s_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

        x_2 = FullyConnectedLayer(name='x_2',
                                  parent=['x_1'],
                                  parent_dim=[x2s_dim],
                                  nout=x2s_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

        rnn = LSTM(name='rnn',
                   parent=['x_2'],
                   parent_dim=[x2s_dim],
                   nout=rnn_dim,
                   unit='tanh',
                   init_W=init_W,
                   init_U=init_U,
                   init_b=init_b)

        theta_1 = FullyConnectedLayer(name='theta_1',
                                      parent=['s_tm1'],
                                      parent_dim=[rnn_dim],
                                      nout=p_x_dim,
                                      unit='relu',
                                      init_W=init_W,
                                      init_b=init_b)

        theta_2 = FullyConnectedLayer(name='theta_2',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=p_x_dim,
                                      unit='relu',
                                      init_W=init_W,
                                      init_b=init_b)

        theta_mu = FullyConnectedLayer(name='theta_mu',
                                       parent=['theta_2'],
                                       parent_dim=[p_x_dim],
                                       nout=target_dim,
                                       unit='linear',
                                       init_W=init_W,
                                       init_b=init_b)

        nodes = [rnn, x_1, x_2, theta_1, theta_2, theta_mu]

        params = OrderedDict()

        for node in nodes:
            if node.initialize() is not None:
                params.update(node.initialize())

        params = init_tparams(params)

        s_0 = rnn.get_init_state(batch_size)

        x_1_temp = x_1.fprop([x], params)
        x_2_temp = x_2.fprop([x_1_temp], params)

        def inner_fn(x_t, s_tm1):

            s_t = rnn.fprop([[x_t], [s_tm1]], params)

            return s_t

        ((s_temp), updates) = \
            theano.scan(fn=inner_fn,
                        sequences=[x_2_temp],
                        outputs_info=[s_0])

        for k, v in updates.iteritems():
            k.default_update = v

        theta_1_temp = theta_1.fprop([s_temp], params)
        theta_2_temp = theta_2.fprop([theta_1_temp], params)
        theta_mu_temp = theta_mu.fprop([theta_2_temp], params)

        recon = MSE(y, theta_mu_temp)
        recon_term = recon.mean()
        recon_term.name = 'recon_term'

        nll_upper_bound = recon_term
        nll_upper_bound.name = 'nll_upper_bound'

        model.inputs = [x, y]
        model.params = params
        model.nodes = nodes

        optimizer = Adam(lr=lr)

        extension = [
            GradientClipping(batch_size=batch_size),
            EpochCount(epoch),
            Picklize(freq=1, force_save_freq=force_saving_freq, path=save_path),
            WeightNorm()
        ]

        mainloop = Training(
            name=pkl_name,
            data=Iterator(train_data, batch_size),
            model=model,
            optimizer=optimizer,
            cost=nll_upper_bound,
            outputs=[nll_upper_bound],
            extension=extension
        )
        mainloop.run()


main()


