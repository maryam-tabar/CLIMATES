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
        pkl_name = 'VRNN_hidden{}'.format(rnn_dim)
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

        z_1 = FullyConnectedLayer(name='z_1',
                                  parent=['z_t'],
                                  parent_dim=[z_dim],
                                  nout=z2s_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

        z_2 = FullyConnectedLayer(name='z_2',
                                  parent=['z_1'],
                                  parent_dim=[z2s_dim],
                                  nout=z2s_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

        rnn = LSTM(name='rnn',
                   parent=['x_2', 'z_2'],
                   parent_dim=[x2s_dim, z2s_dim],
                   nout=rnn_dim,
                   unit='tanh',
                   init_W=init_W,
                   init_U=init_U,
                   init_b=init_b)

        phi_1 = FullyConnectedLayer(name='phi_1',
                                    parent=['x_2', 's_tm1'],
                                    parent_dim=[x2s_dim, rnn_dim],
                                    nout=q_z_dim,
                                    unit='relu',
                                    init_W=init_W,
                                    init_b=init_b)

        phi_2 = FullyConnectedLayer(name='phi_2',
                                    parent=['phi_1'],
                                    parent_dim=[q_z_dim],
                                    nout=q_z_dim,
                                    unit='relu',
                                    init_W=init_W,
                                    init_b=init_b)

        phi_mu = FullyConnectedLayer(name='phi_mu',
                                     parent=['phi_2'],
                                     parent_dim=[q_z_dim],
                                     nout=z_dim,
                                     unit='linear',
                                     init_W=init_W,
                                     init_b=init_b)

        phi_sig = FullyConnectedLayer(name='phi_sig',
                                      parent=['phi_2'],
                                      parent_dim=[q_z_dim],
                                      nout=z_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

        prior_1 = FullyConnectedLayer(name='prior_1',
                                      parent=['s_tm1'],
                                      parent_dim=[rnn_dim],
                                      nout=p_z_dim,
                                      unit='relu',
                                      init_W=init_W,
                                      init_b=init_b)

        prior_2 = FullyConnectedLayer(name='prior_2',
                                      parent=['prior_1'],
                                      parent_dim=[p_z_dim],
                                      nout=p_z_dim,
                                      unit='relu',
                                      init_W=init_W,
                                      init_b=init_b)

        prior_mu = FullyConnectedLayer(name='prior_mu',
                                       parent=['prior_2'],
                                       parent_dim=[p_z_dim],
                                       nout=z_dim,
                                       unit='linear',
                                       init_W=init_W,
                                       init_b=init_b)

        prior_sig = FullyConnectedLayer(name='prior_sig',
                                        parent=['prior_2'],
                                        parent_dim=[p_z_dim],
                                        nout=z_dim,
                                        unit='softplus',
                                        cons=1e-4,
                                        init_W=init_W,
                                        init_b=init_b_sig)

        theta_1 = FullyConnectedLayer(name='theta_1',
                                      parent=['z_2', 's_tm1'],
                                      parent_dim=[z2s_dim, rnn_dim],
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

        theta_sig = FullyConnectedLayer(name='theta_sig',
                                        parent=['theta_2'],
                                        parent_dim=[p_x_dim],
                                        nout=target_dim,
                                        unit='softplus',
                                        cons=1e-4,
                                        init_W=init_W,
                                        init_b=init_b_sig)

        nodes = [rnn,
                 x_1, x_2, z_1, z_2,
                 phi_1, phi_2, phi_mu, phi_sig,
                 prior_1, prior_2, prior_mu, prior_sig,
                 theta_1, theta_2, theta_mu, theta_sig]

        params = OrderedDict()

        for node in nodes:
            if node.initialize() is not None:
                params.update(node.initialize())

        params = init_tparams(params)

        s_0 = rnn.get_init_state(batch_size)

        x_1_temp = x_1.fprop([y], params)
        x_2_temp = x_2.fprop([x_1_temp], params)

        def inner_fn(x_t, s_tm1):

            phi_1_t = phi_1.fprop([x_t, s_tm1], params)
            phi_2_t = phi_2.fprop([phi_1_t], params)
            phi_mu_t = phi_mu.fprop([phi_2_t], params)
            phi_sig_t = phi_sig.fprop([phi_2_t], params)

            prior_1_t = prior_1.fprop([s_tm1], params)
            prior_2_t = prior_2.fprop([prior_1_t], params)
            prior_mu_t = prior_mu.fprop([prior_2_t], params)
            prior_sig_t = prior_sig.fprop([prior_2_t], params)

            z_t = Gaussian_sample(phi_mu_t, phi_sig_t)
            z_1_t = z_1.fprop([z_t], params)
            z_2_t = z_2.fprop([z_1_t], params)

            s_t = rnn.fprop([[x_t, z_2_t], [s_tm1]], params)

            return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_2_t

        ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, z_2_temp), updates) = \
            theano.scan(fn=inner_fn,
                        sequences=[x_2_temp],
                        outputs_info=[s_0, None, None, None, None, None])

        for k, v in updates.iteritems():
            k.default_update = v

        s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)
        theta_1_temp = theta_1.fprop([z_2_temp, s_temp], params)
        theta_2_temp = theta_2.fprop([theta_1_temp], params)
        theta_mu_temp = theta_mu.fprop([theta_2_temp], params)
        theta_sig_temp = theta_sig.fprop([theta_2_temp], params)

        recon = Gaussian(y, theta_mu_temp, theta_sig_temp)
        recon_term = recon.mean()
        recon_term.name = 'recon_term'

        kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)
        kl_term = kl_temp.mean()
        kl_term.name = 'kl_term'

        nll_upper_bound = recon_term + kl_term
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




