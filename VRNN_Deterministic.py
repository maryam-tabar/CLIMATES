#### The code is adapted from https://github.com/jych/nips2015_vrnn which is linked to the following reference.
##Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A.C., Bengio, Y.: A recurrent latent variable model for sequential data. Advances in neural information processing systems 28, 2980–2988 (2015)
##
import os

os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
os.environ['KERAS_BACKEND'] = 'theano'
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
from cle.cle.train.opt import Adam
from cle.cle.utils import init_tparams, sharedX
from cle.cle.utils.compat import OrderedDict
from cle.cle.utils.op import Gaussian_sample
from cle.cle.utils.gpu_op import concatenate
from cle.cle.utils import unpickle

from datasets.npp import NPP


def main():
    for rnn_dim in [256]:
        for count in np.arange(500):
            pkl_name = 'VRNN_hidden{}_{}'.format(rnn_dim, count)
            npp_path = './datasets/npp.npy'
            save_path = './models/'
            monitoring_freq = 8
            force_saving_freq = 8
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

            if count == 0:
                pkl_path = './models/VRNN_hidden{}.pkl'.format(rnn_dim)
            else:
                pkl_path = './models/VRNN_hidden{}_{}.pkl'.format(rnn_dim, (count - 1))

            exp = unpickle(pkl_path)
            model = exp.model
            nodes = exp.model.nodes
            names = [node.name for node in nodes]
            params = exp.model.params
            [x, y] = exp.model.inputs

            [rnn,
             x_1, x_2, z_1, z_2,
             phi_1, phi_2, phi_mu, phi_sig,
             prior_1, prior_2, prior_mu, prior_sig,
             theta_1, theta_2, theta_mu] = nodes

            train_data = NPP(name='train', path=npp_path)

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

            recon = MSE(y, theta_mu_temp)
            recon_term = recon.mean()
            recon_term.name = 'recon_term'

            kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)
            kl_term = kl_temp.mean()
            kl_term.name = 'kl_term'

            nll_upper_bound = recon_term + kl_term
            nll_upper_bound.name = 'nll_upper_bound'

            optimizer = Adam(lr=lr)

            extension = [
                GradientClipping(batch_size=batch_size),
                EpochCount(epoch),
                Picklize(freq=monitoring_freq, force_save_freq=force_saving_freq, path=save_path),
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
