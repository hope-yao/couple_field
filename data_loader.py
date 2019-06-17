import numpy as np
import scipy.io as sio


def load_data(snr=None, percent=0.0):
    num_node = 37
    # Purely thermal
    # data = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_data5.mat')

    # purely structural
    #data = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_data2.mat')
    data0 = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_data_half_loading_noTem.mat')
    #data = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_data_half_loading_x.mat')
    #data = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_data_half_loading_y.mat')

    # coupled loading
    data1 = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_data4.mat')
    data2 = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_all_loading.mat')
    data3 = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_data_half_loading.mat')
    data4 = sio.loadmat('2D_thermoelastic_36by36_xy_fixed_single_line_loading.mat')

    data = data2
    train_load = np.expand_dims(np.stack([-data['fx'].astype('float64')/1e5,
                                    -data['fy'].astype('float64')/1e5,
                                    data['ftem'].astype('float64')], -1), 0)
    train_resp = np.expand_dims(np.stack([data['ux'].astype('float64')*1e4,
                                    data['uy'].astype('float64')*1e4,
                                    data['utem'].astype('float64')], -1), 0)

    data = data4
    test_load = np.expand_dims(np.stack([-data['fx'].astype('float64')/1e5,
                                    -data['fy'].astype('float64')/1e5,
                                    data['ftem'].astype('float64')], -1), 0)
    test_resp = np.expand_dims(np.stack([data['ux'].astype('float64')*1e4,
                                    data['uy'].astype('float64')*1e4,
                                    data['utem'].astype('float64')], -1), 0)

    if percent is not None:
        train_loading_w_noise = np.zeros_like(train_load)
        train_response_w_noise = np.zeros_like(train_resp)
        for i in range(train_load.shape[0]):
            for j in range(train_load.shape[-1]):
                np.random.seed(0)
                noise = percent * np.random.normal(size=train_load.shape, scale=1)
                train_loading_w_noise = (1) * train_load #+ noise
                np.random.seed(1)
                percent = train_load * 0.01
                noise = percent * np.random.normal(size=train_load.shape, scale=1)#np.random.uniform(0,1,size=train_load.shape)#
                train_response_w_noise = (1) * train_resp  + noise

        for i in range(test_load.shape[0]):
            for j in range(test_load.shape[-1]):
                np.random.seed(2)
                noise = 0#percent * np.random.normal(size=test_load.shape)
                test_loading_w_noise = (1+noise) * test_load
                np.random.seed(3)
                noise = 0#percent * np.random.normal(size=test_load.shape)
                test_response_w_noise = (1+noise) * test_resp

    elif snr is not None:
        loading_w_noise = np.zeros_like(load)
        response_w_noise = np.zeros_like(resp)
        for i in range(load.shape[0]):
            for j in range(load.shape[-1]):
                low_val = load.min() / 10 ** (snr / 20)
                max_val = load.max() / 10 ** (snr / 20)
                np.random.seed(4)
                noise = np.random.uniform(low=low_val, high=max_val, size=(load.shape[1:3]))
                loading_w_noise[i, :, :, j] = noise + load[i, :, :, j]
                low_val = resp.min() / 10 ** (snr / 20)
                max_val = resp.max() / 10 ** (snr / 20)
                np.random.seed(5)
                noise = np.random.uniform(low=low_val, high=max_val, size=(resp.shape[1:3]))
                response_w_noise[i, :, :, j] = noise + resp[i, :, :, j]
        # resp = response_w_noise
        # load = loading_w_noise

    rho = [212e0, 0.288, 16., 1.2e-4] # E, mu, k, alpha

    data = {'num_node': num_node,
            'rho': rho,
            'train_load': train_loading_w_noise,
            'train_resp': train_response_w_noise,
            'clean_train_load': train_load,
            'clean_train_resp': train_resp,
            'test_load': test_loading_w_noise,
            'test_resp': test_response_w_noise,
            }

    # num_node = 3
    # data = {'num_node': num_node,
    #         'rho': rho,
    #         'train_load': train_load[:,17:20,17:20,:],
    #         'train_resp': train_resp[:,17:20,17:20,:],
    #         'test_load': test_load[:,17:20,17:20,:],
    #         'test_resp': test_resp[:,17:20,17:20,:],
    #         }

    return data
