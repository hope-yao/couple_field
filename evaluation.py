import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import *

class Evaluator(object):
    def __init__(self, model, data):
        self.model = model

        self.data = data
        self.init_w = np.zeros((3,3,1,1))

        self.loss_value = None
        self.grads_value = None

        self.loss_tf = self.model.get_loss()
        self.hessian_tf = self.model.get_hessian()
        self.grad_tf = self.model.get_grad()
        self.initial_graph()

    def initial_graph(self):
        # initialize
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_loss(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.loss_value = self.sess.run(self.loss_tf, self.feed_dict).astype('float64')
        return self.loss_value

    def get_grads(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.grads_value = self.sess.run(self.grad_tf, self.feed_dict)[0].flatten().astype('float64')
        return self.grads_value

    def get_hessian(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.hessian_value = self.sess.run(self.hessian_tf, self.feed_dict)[0].astype('float64')
        return self.hessian_value

    def get_pred_load(self,w):
        feed_dict = {#self.model.load_pl: data['train_load'],
                      self.model.resp_pl: data['train_resp'],
                      self.model.trainable_var_pl: w.astype('float64')}
        pred_value = self.sess.run(self.model.load_pred, feed_dict)
        return pred_value

    def run_BFGS(self):
        from scipy.optimize import fmin_l_bfgs_b
        x, min_val, info = fmin_l_bfgs_b(self.get_loss, self.init_w.flatten(),
                                         fprime=self.get_grads, maxiter=200, maxfun=200,
                                         disp= True)
        print('    loss: {}'.format(min_val))
        pass

    def run_newton(self):
        from scipy.optimize import minimize
        self.result = minimize(self.get_loss, self.model.trainable_var_np, method='Newton-CG',
                          jac=self.get_grads, hess=self.get_hessian,
                          options={'xtol': 1e-25, 'disp': True})
        return self.result.x

    def run_trust_ncg(self):
        from scipy.optimize import minimize
        self.result = minimize(self.get_loss, self.model.trainable_var_np, method='trust-ncg',
                          jac=self.get_grads, hess=self.get_hessian,
                          options={'gtol': 1e-3, 'disp': True})
        return self.result.x

    def run_tnc(self):
        from scipy.optimize import fmin_tnc

        self.result = fmin_tnc(self.get_loss,
                               self.model.trainable_var_np,
                               fprime=self.get_grads,
                               stepmx=100,
                               pgtol=1e-8,
                               # ftol=1e-15,
                               maxfun=20000,
                               disp='True')

        return self.result[0]

    def run_least_squares(self):
        # problematic
        from scipy.optimize import least_squares
        self.result = least_squares(self.get_loss,
                               self.model.trainable_var_np,
                               jac=self.get_grads,

                               method='trf',
                               gtol=1e-5,
                               # ftol=1e-15,
                               loss='linear',
                               verbose=2)

        return self.result[0]

    def init_solve(self, load, omega=2/3.):
        self.model.init_solve(load, omega)
        self.solution = {'itr':[], 'loss': [], 'pred':[]}

    def run_forward(self, filter, pred_i, resp_ref=None, max_itr=100):

        st = 0 if self.solution['itr'] == [] else self.solution['itr'][-1]+10
        for itr in tqdm(range(st, st+max_itr, 1)):
            feed_dict = {self.model.u_in: pred_i, self.model.trainable_var_pl:filter}
            pred_i = self.sess.run(self.model.u_out, feed_dict)
            if itr%100 == 0:
                self.solution['itr'] += [itr]
                self.solution['pred'] += [pred_i]
                if resp_ref is not None:
                    pred_err_i = relative_l2_err(pred_i, resp_ref)
                    print("iter:{}  pred_err: {}".format(itr, np.mean(pred_err_i)))
                    self.solution['loss'] += [np.mean(pred_err_i)]

        return pred_i

def relative_l2_err(a,b):
    # a:pred, b:reference
    return np.sqrt(np.sum((a - b) ** 2)) / np.sqrt(np.sum((b) ** 2))


def visualization(evaluator, data):

    evaluator.init_solve(load=data['test_load'], omega=2/3.)
    pred_i = np.zeros_like(data['test_resp'])  # data['test_resp']#
    resp_ref = data['test_resp']
    pred_resp_ref = evaluator.run_forward(model.trainable_var_ref, pred_i, resp_ref, max_itr=4000)
    s0 = evaluator.solution

    # test the model
    evaluator.init_solve(load=data['test_load'], omega=2/3.)
    pred_i = np.zeros_like(data['test_resp'])  # data['test_resp']#
    pred_resp = evaluator.run_forward(result, pred_i, resp_ref, max_itr=4000)
    s1 = evaluator.solution

    plt.figure()
    plt.semilogy(s0['itr'], s0['loss'], label='ref')
    plt.semilogy(s1['itr'], s1['loss'], label='pred')
    plt.legend()

    pred_load = evaluator.get_pred_load(result)

    plt.figure(figsize=(6, 6))
    idx = 0  # which data to visualize
    for i in range(3):
        plt.subplot(6, 3, i + 1)
        plt.imshow(data['train_load'][idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 3 + i + 1)
        plt.imshow(data['clean_train_resp'][idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 6 + i + 1)
        plt.imshow(data['train_resp'][idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 9 + i + 1)
        plt.imshow(data['clean_train_resp'][idx, 1:-1, 1:-1, i]-data['train_resp'][idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 12 + i + 1)
        plt.imshow(pred_load[idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 15 + i + 1)
        plt.imshow(data['train_load'][idx, 1:-1, 1:-1, i] - pred_load[idx, 1:-1, 1:-1, i])
        plt.colorbar()

    plt.figure(figsize=(6, 6))
    idx = 0  # which data to visualize
    for i in range(3):
        plt.subplot(6, 3, i + 1)
        plt.imshow(data['test_load'][idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 3 + i + 1)
        plt.imshow(data['test_resp'][idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 6 + i + 1)
        plt.imshow(pred_resp[idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 9 + i + 1)
        plt.imshow(pred_resp_ref[idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 12 + i + 1)
        plt.imshow(pred_resp_ref[idx, 1:-1, 1:-1, i]-data['test_resp'][idx, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(6, 3, 15 + i + 1)
        plt.imshow(pred_resp[idx, 1:-1, 1:-1, i]-data['test_resp'][idx, 1:-1, 1:-1, i])
        plt.colorbar()

    import seaborn as sns
    plt.figure()
    for i in range(3):
        plt.subplot(2,3,i+1)
        sns.distplot((data['train_load'][idx, 1:-1, 1:-1, i] - pred_load[idx, 1:-1, 1:-1, i]).flatten())
        plt.subplot(2,3,i+1+3)
        sns.distplot((data['clean_train_resp'][idx, 1:-1, 1:-1, i] - data['train_resp'][idx, 1:-1, 1:-1, i]).flatten())


    for i in range(2):
        mat = result[9*i:9*(i+1)]
        print(mat.reshape(3,3))
        print(np.sum(mat))
    print(model.w_ref['xt'].reshape(3,3))
    print(np.sum(model.w_ref['xt']))
    print(model.w_ref['yt'].reshape(3,3))
    print(np.sum(model.w_ref['yt']))
    plt.show()


if __name__ == "__main__":
    from therm_elast_model import * # not unique
    # from therm_elast_train_9filters_v2 import * # not unique

    cfg = {'all_para': ['xx','xy','xt',
                        'yx','yy','yt',
                        'tx','ty','tt'], # DO NOT CHANGE ORDER
           'unknown_para': ['xx','xy','xt','yx', 'yy', 'yt', 'tx', 'ty', 'tt'], # (1) with random init, converge to something different, inference diverge (2)takes very long with zero init
           # 'unknown_para': ['xy','xt','yx', 'yt', 'tx', 'ty'], # off diagonal, (1) converges with random init. (2)takes very long with zero init, converging but not to reference solution
           # 'unknown_para': ['xx', 'yy', 'tt'], # diagonal, converge to reference solution
           }

    # load data
    data = load_data(percent=0.000)#snr=100#

    # build the network
    model = FEA_Net_h(data,cfg)

    # train the network
    evaluator = Evaluator(model, data)
    result = evaluator.run_newton()#run_trust_ncg
    # evaluator.sess.run(model.w_tf['yy'], {model.trainable_var_pl: result}).tolist()
    # evaluator.sess.run(model.loss, {model.trainable_var_pl: model.trainable_var_ref, model.load_pl: data['train_load'],  model.resp_pl: data['train_resp']})

    pred1 = evaluator.get_pred_load(result)
    pred2 = evaluator.get_pred_load(model.trainable_var_ref)
    for para_i in cfg['all_para']:
        w_est = evaluator.sess.run(model.w_tf[para_i], {model.trainable_var_pl:result})
        w_ref = model.w_ref[para_i]
        print(np.squeeze(w_est))
        print(np.squeeze(w_ref))
        print(np.sum(w_est),np.sum(w_ref))
        err = np.mean((w_est - w_ref) ** 2) / np.mean(w_ref ** 2)
        if para_i in cfg['unknown_para']:
            print('{}, unknown, err: {}'.format(para_i, err))
        else:
            print('{}, known, err: {}'.format(para_i, err))

    plt.figure(figsize=(10,3))
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.imshow(pred1[0, 1:-1, 1:-1, i])
        plt.colorbar()
        plt.subplot(2, 3, 3+ i + 1)
        plt.imshow(pred1[0, 1:-1, 1:-1, i] - pred2[0, 1:-1, 1:-1, i])
        plt.colorbar()
    plt.show()

    # visualize training result
    visualization(evaluator,data)
    pass
