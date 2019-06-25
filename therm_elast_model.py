import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


class FEA_Net_h():
    # NOTICE: right now for homogeneous anisotropic material only!!
    def __init__(self, data, cfg):
        self.cfg = cfg
        # self.batch_size = 4

        # data related
        self.num_node = data['num_node']
        self.E, self.mu, self.k, self.alpha = self.rho = data['rho'] #

        # 3 dimensional in and out, defined on the nodes
        self.load_pl = tf.placeholder(tf.float64, shape=(None, data['num_node'], data['num_node'], 3), name='load_pl')
        self.resp_pl = tf.placeholder(tf.float64, shape=(None, data['num_node'], data['num_node'], 3), name='resp_pl')

        # get filters
        self.get_w_matrix()
        self.load_pred = self.u2v_map()


    def get_w_matrix(self):
        self.w_ref = {}
        self.get_w_matrix_elast()
        self.get_w_matrix_thermal()
        self.get_w_matrix_coupling()
        self.apply_physics_constrain()

    def apply_physics_constrain(self):
        self.trainable_var_pl = tf.placeholder(tf.float64, shape=(9 * len(self.cfg['unknown_para']),), name='filter_vector')
        w_tf_unknown = tf.split(self.trainable_var_pl, len(self.cfg['unknown_para']))

        self.w_tf = {}
        self.trainable_var_np = []
        self.trainable_var_ref = []
        self.singula_penalty = 0
        cnt = 0
        for para_i in self.cfg['all_para']:
            if para_i in self.cfg['unknown_para']:
                # unknown physics
                if 1:
                    np.random.seed(111*cnt) # problem with multi-col-linearility will converge differently with different initial seed.
                    self.trainable_var_np += [10*np.random.randn(*self.w_ref[para_i].shape).flatten()] # initial guess with random number
                else:
                    self.trainable_var_np += [np.zeros_like(self.w_ref[para_i]).flatten()] # initial guess with all zeros
                self.trainable_var_ref += [self.w_ref[para_i].flatten()] # reference solution, ground truth
                self.w_tf[para_i] = tf.reshape(w_tf_unknown[cnt], (3,3,1,1)) # placeholder for optimized input
                cnt += 1
                self.singula_penalty += tf.abs(tf.reduce_sum(self.w_tf[para_i]))
            else:
                # known physics
                self.w_tf[para_i] = tf.constant(self.w_ref[para_i])

        wxx_diag = tf.abs(self.w_tf['xx'][1:2,1:2])
        self.wxx_off_diag_max = tf.reduce_max([tf.abs(self.w_tf['xx']), tf.abs(self.w_tf['xy']), tf.abs(self.w_tf['xt'])])
        wyy_diag = tf.abs(self.w_tf['yy'][1:2,1:2])
        self.wyy_off_diag_max = tf.reduce_max([tf.abs(self.w_tf['yy']), tf.abs(self.w_tf['yx']), tf.abs(self.w_tf['yt'])])
        wtt_diag = self.w_tf['tt'][1:2,1:2]
        self.wtt_off_diag_max = tf.reduce_max([tf.abs(self.w_tf['tt']), tf.abs(self.w_tf['tx']), tf.abs(self.w_tf['ty'])])
        self.diagonal_dominant_penalty = tf.exp(tf.reduce_mean(self.wxx_off_diag_max - wxx_diag)) \
                                         + tf.exp(tf.reduce_mean(self.wyy_off_diag_max - wyy_diag)) \
                                         + tf.exp(tf.reduce_mean(self.wtt_off_diag_max - wtt_diag))

        self.trainable_var_np = np.concatenate(self.trainable_var_np,0)
        self.trainable_var_ref = np.concatenate(self.trainable_var_ref,0)

        # tf.nn.conv2d filter shape: [filter_height, filter_width, in_channels, out_channels]
        self.w_filter = tf.concat([tf.concat([self.w_tf['xx'], self.w_tf['xy'], self.w_tf['xt']],2),
                                   tf.concat([self.w_tf['yx'], self.w_tf['yy'], self.w_tf['yt']],2),
                                   tf.concat([self.w_tf['tx'], self.w_tf['ty'], self.w_tf['tt']],2)],
                                  3)
        self.w_filter_ref = tf.concat([tf.concat([self.w_ref['xx'], self.w_ref['xy'], self.w_ref['xt']],2),
                                   tf.concat([self.w_ref['yx'], self.w_ref['yy'], self.w_ref['yt']],2),
                                   tf.concat([self.w_ref['tx'], self.w_ref['ty'], self.w_ref['tt']],2)],
                                  3)

    def get_w_matrix_coupling(self):
        E, v = self.E, self.mu
        alpha = self.alpha
        self.w_ref['tx'] = np.zeros((3,3,1,1), dtype='float64')
        self.w_ref['ty'] = np.zeros((3,3,1,1), dtype='float64')
        coef = E * alpha / (6*(v-1)) / 400 *1e4
        self.w_ref['xt'] = coef * np.asarray([[1, 0, -1],
                                      [4, 0, -4],
                                      [1, 0, -1]]
                                     , dtype='float64').reshape(3,3,1,1)

        self.w_ref['yt'] = coef * np.asarray([[-1, -4, -1],
                                      [0, 0, 0],
                                      [1, 4, 1]]
                                     , dtype='float64').reshape(3,3,1,1)

    def get_w_matrix_thermal(self):
        w = -1/3. * self.k * np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
        w = np.asarray(w, dtype='float64')
        self.w_ref['tt'] = w.reshape(3,3,1,1)

    def get_w_matrix_elast(self):
        E, mu = self.E, self.mu
        if 0:
            cost_coef = E / 16. / (1 - mu ** 2)
            wxx = cost_coef * np.asarray([
                [-4 * (1 - mu / 3.), 16 * mu / 3., -4 * (1 - mu / 3.)],
                [-8 * (1 + mu / 3.), 32. * (1 - mu / 3.), -8 * (1 + mu / 3.)],
                [-4 * (1 - mu / 3.), 16 * mu / 3., -4 * (1 - mu / 3.)],
            ], dtype='float64')

            wxy = wyx = cost_coef * np.asarray([
                [2 * (mu + 1), 0, -2 * (mu + 1)],
                [0, 0, 0],
                [-2 * (mu + 1), 0, 2 * (mu + 1)],
            ], dtype='float64')

            wyy = cost_coef * np.asarray([
                [-4 * (1 - mu / 3.), -8 * (1 + mu / 3.), -4 * (1 - mu / 3.)],
                [16 * mu / 3., 32. * (1 - mu / 3.), 16 * mu / 3.],
                [-4 * (1 - mu / 3.), -8 * (1 + mu / 3.), -4 * (1 - mu / 3.)],
            ], dtype='float64')
        else:
            wxx = E / 4. / (1 - mu**2) * np.asarray([
                [-(1 - mu / 3.), 4* mu / 3., -(1 - mu / 3.)],
                [-2*(1 + mu / 3.), 8*(1 - mu / 3.), -2*(1 + mu / 3.)],
                [-(1 - mu / 3.), 4* mu / 3., -(1 - mu / 3.)],
            ], dtype='float64')
            wyy = E / 4. / (1 - mu**2) * np.asarray([
                [-(1 - mu / 3.), -2*(1 + mu / 3.), -(1 - mu / 3.)],
                [4* mu / 3., 8*(1 - mu / 3.), 4* mu / 3.],
                [-(1 - mu / 3.), -2*(1 + mu / 3.), -(1 - mu / 3.)],
            ], dtype='float64')
            wxy = wyx = E / 8. / (1 - mu) * np.asarray([
                [1, 0, -1],
                [0, 0, 0],
                [-1, 0, 1],
            ], dtype='float64')

        self.w_ref['xx'] = wxx.reshape(3,3,1,1)
        self.w_ref['xy'] = wxy.reshape(3,3,1,1)
        self.w_ref['yx'] = wyx.reshape(3,3,1,1)
        self.w_ref['yy'] = wyy.reshape(3,3,1,1)

    def boundary_padding(self,x):
        ''' special symmetric boundary padding '''
        left = x[:, :, 1:2, :]
        right = x[:, :, -2:-1, :]
        upper = tf.concat([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
        down = tf.concat([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
        padded_x = tf.concat([left, x, right], 2)
        padded_x = tf.concat([upper, padded_x, down], 1)
        return padded_x

    def u2v_map(self):
        padded_resp = self.boundary_padding(self.resp_pl)  # for boundary consideration
        wx = tf.nn.conv2d(input=padded_resp, filter=self.w_filter, strides=[1, 1, 1, 1], padding='VALID')
        return wx

    def get_loss(self):
        self.diff = self.load_pred - self.load_pl
        diff_not_on_bc = self.apply_bc(self.diff)#[:,:,:,0]
        self.l1_error = tf.reduce_mean(diff_not_on_bc**2)
        # self.l1_error = tf.reduce_mean((diff_not_on_bc*self.apply_bc(self.resp_pl))**2)
        self.loss = self.l1_error  #+ 0.1*self.diagonal_dominant_penalty #+ 1*self.singula_penalty
        return self.loss

    def get_grad(self):
        self.rho_grads = tf.gradients(self.loss, self.trainable_var_pl)
        return self.rho_grads

    def get_hessian(self):
        self.rho_hessian = tf.hessians(self.loss, self.trainable_var_pl)
        return self.rho_hessian

    # V2U mapping functions
    def apply_bc(self, x):
        x_bc = tf.pad(x[:, 1:-1, 1:-1, :], ((0,0), (1, 1),(1, 1), (0, 0)), "constant")  # for boundary consideration
        return x_bc

    def FEA_conv(self, w, x):
        padded_input = self.boundary_padding(x)  # for boundary consideration
        wx = tf.nn.conv2d(input=padded_input, filter=w, strides=[1, 1, 1, 1], padding='VALID')
        wx_bc = wx * self.bc_mask # boundary_corrrect
        return wx_bc

    def v2u_layer(self, w, x):
        wx = self.FEA_conv(w, x)
        wx_bc = self.apply_bc(wx)
        return wx_bc

    def get_dmat(self):
        d_matrix = tf.stack([self.w_tf['xx'][1,1,0,0], self.w_tf['yy'][1,1,0,0], self.w_tf['tt'][1,1,0,0]])  # x, y, and t components
        return tf.reshape(d_matrix,(1,1,1,3))

    def get_bc_mask(self):
        bc_mask = np.ones_like(self.new_load)
        bc_mask[:, 0, :, :] /= 2
        bc_mask[:, -1, :, :] /= 2
        bc_mask[:, :, 0, :] /= 2
        bc_mask[:, :, -1, :] /= 2
        return bc_mask

    def init_solve(self, load, omega):
        self.omega = omega
        self.new_load = load
        self.d_matrix = self.get_dmat()
        self.bc_mask = self.get_bc_mask()
        self.u_in = tf.placeholder(tf.float64, load.shape, name='u_in')
        self.u_out = self.apply(self.u_in)

    def apply(self, u_in):
        wx = self.v2u_layer(self.w_filter, u_in)
        u_out = self.omega * (self.new_load - wx) / self.d_matrix +  u_in
        return u_out