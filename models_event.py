import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def init_gaussian_tuning(n, m, x_min:np.array, x_max:np.array):
    '''
    :param n: 特征的数量，int
    :param m: 编码一个特征所使用的神经元数量，int
    :param x_min: n个特征的最小值，shape=[n]的tensor
    :param x_max: n个特征的最大值，shape=[n]的tensor
    '''
    assert(len(x_min.shape) == 1 and x_min.shape[0] == n)
    assert(len(x_max.shape) == 1 and x_max.shape[0] == n)
    assert m > 2
    i = np.tile(np.expand_dims(np.linspace(1, m, m), axis=0), [n, 1]).astype(np.float32)
    x_min_ex = np.tile(np.expand_dims(x_min, axis=-1), [1, m]) # [n, m]
    x_max_ex = np.tile(np.expand_dims(x_max, axis=-1), [1, m]) # [n, m]
    mu = (2 * i - 3) / 2 * (x_max_ex - x_min_ex) / (m - 2) # [n, m]
    sigma2 = np.power(np.expand_dims(1 / 1.5 * (x_max - x_min) / (m - 2), -1), 2) # [n, 1]
    sigma2 = np.tile(sigma2, [1, m]) # [n, m]
    return n, m, mu, sigma2

def gaussian_encode_ori_convert_to_dagger_fail(x: np.array, k,  n, m, mu, sigma2, max_spike_time=50):
    '''
    :param x: shape=[batch_size, n, k]，batch_size个数据，每个数据含有n个特征，每个特征中有k个数据
    :param max_spike_time: 最大（最晚）脉冲发放时间，也可以称为编码时间窗口的长度
    :return: out_spikes, shape=[batch_size, n, k, m]，将每个数据编码成了m个神经元的脉冲发放时间
    '''
    x_shape = tf.shape(x, name='x_shape') # [batch_size, n, k]
    x = tf.transpose(x, [0, 2, 1], name='x_transpose') # [batch_size, n, k] -> [batch_size, k, n]
    x = tf.reshape(x, [x_shape[0]*k, n], name='x_reshape') # shape=[batch_size * k, n]
    x = tf.expand_dims(x, 2, name='x_expand') # shape=[batch_size * k, n, 1]
    x = tf.tile(x, [1,1,m], name='x_tile') # shape=[batch_size * k, n, m]
    x = tf.square(x - mu, name='x_square')
    y = tf.exp(- x / 2 / sigma2) # shape=[batch_size * k, n, m]
    x = -1 * tf.cast(y<=0.5/max_spike_time, tf.float32) # use 0.5/max_spike_time to substitude round
    x += tf.floor(max_spike_time * (1 - y)+0.5, name='spikes_round') * tf.cast(y>0.5/max_spike_time, tf.float32) # [batch_size * k, n, m]
    x = tf.reshape(x, [x_shape[0], k, n, m], name='out_spikes_reshape') # shape: [batch_size * k, n, m] -> [batch_size, k, n, m]
    x = tf.transpose(x, [0,2,1,3], name='out_spikes_transpose') # shape: [batch_size, k, n, m] -> [batch_size, n, k, m]

def gaussian_encode(x: np.array, k,  n, m, mu, sigma2, max_spike_time=50):
    '''
    :param x: shape=[batch_size, n, k]，batch_size个数据，每个数据含有n个特征，每个特征中有k个数据
    :param max_spike_time: 最大（最晚）脉冲发放时间，也可以称为编码时间窗口的长度
    :return: out_spikes, shape=[batch_size, n, k, m]，将每个数据编码成了m个神经元的脉冲发放时间
    '''
    x = tf.transpose(x, [0, 2, 1], name='x_transpose') # [batch_size, n, k] -> [batch_size, k, n]
    x = tf.expand_dims(x, -1, name='x_expand') # shape=[batch_size， k, n, 1]
    x = tf.tile(x, [1, 1,1,m], name='x_tile') # shape=[batch_size， k, n, m]
    x = tf.square(x - mu, name='x_square')
    y = tf.exp(- x / 2 / sigma2) # shape=[batch_size， k, n, m]
    x = -1 * tf.cast(y<=0.5/max_spike_time, tf.float32) # use 0.5/max_spike_time to substitude round
    x += tf.floor(max_spike_time * (1 - y)+0.5, name='spikes_round') * tf.cast(y>0.5/max_spike_time, tf.float32) # [batch_size, k, n, m]
    x = tf.transpose(x, [0,2,1,3], name='out_spikes_transpose') # shape: [batch_size, k, n, m] -> [batch_size, n, k, m]
    return x

def init_net(tau=15.0, tau_s=15.0 / 4, v_threshold=1.0):
        tau = np.float32(tau)
        tau_s = np.float32(tau_s)
        v_threshold = np.float32(v_threshold)
        t_max = (tau * tau_s * np.log(tau / tau_s)) / (tau - tau_s)
        v0 = v_threshold / (np.exp(-t_max / tau) - np.exp(-t_max / tau_s))
        return t_max, v0, tau, tau_s

def net_ori_convert_to_dagger_fail(in_spikes, in_feature, out_feature, T, t_max, v0, tau, tau_s, ret_type, is_training, reuse=None, scope='fcEventNet'):
    def psp_kernel(t: tf.Tensor, tau, tau_s):
        '''
        :param t: 表示时刻的tensor
        :param tau: LIF神经元的积分时间常数
        :param tau_s: 突触上的电流的衰减时间常数
        :return: t时刻突触后的LIF神经元的电压值
        '''
        # 指数衰减的脉冲输入
        return (tf.exp(-t / tau) - tf.exp(-t / tau_s)) * tf.cast(t >= 0, tf.float32)
    # inspikes: [batch_size, in_features]
    t = np.arange(T, dtype=np.float32) # t = [0, 1, 2, ..., T-1] shape=[T]
    t = np.reshape(t, [1,1,t.shape[0]]) # [1,1,T]
    t = tf.tile(t, [tf.shape(in_spikes)[0], in_feature, 1]) # shape=[batch_size, in_features, T]
    in_spikes = tf.tile(tf.expand_dims(in_spikes, -1), [1,1,T], 'tile0') # shape=[batch_size, in_features, T]
    in_spikes = v0 * psp_kernel(t - in_spikes, tau, tau_s) * tf.cast(in_spikes >= 0, tf.float32) # shape=[batch_size, in_features, T]
    in_spikes = tf.transpose(in_spikes, [0, 2, 1], name='trans0') # shape=[batch_size, T, in_features]
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=None,
                        biases_initializer = None,
                        normalizer_fn=None):
        with tf.variable_scope(scope, 'fcEventNet', [in_spikes], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
                    
                    v_out = slim.fully_connected(in_spikes, 64, activation_fn=None, scope='fc0') # shape=[batch_size, T, 64]
                    v_out = slim.fully_connected(v_out, 32, activation_fn=None, scope='fc1') # shape=[batch_size, T, 32]
                    v_out = slim.fully_connected(v_out, out_feature, activation_fn=None, scope='fc2') # shape=[batch_size, T, out_features]

    if ret_type == 'v_max':
        v_out = tf.nn.max_pool1d(v_out, T, 1, 'VALID', data_format='NWC', name='v_out_global_max_pool') # shape=[batch_size, 1, out_features]
    else:
        raise NotImplementedError('not implemented')
    v_out = tf.squeeze(v_out, 1, name='logits') # shape=[batch_size, out_features]
    return v_out

def net(in_spikes, in_feature, out_feature, T, t_max, v0, tau, tau_s, ret_type, is_training, reuse=None, scope='fcEventNet'):
    def psp_kernel(t: tf.Tensor, tau, tau_s):
        '''
        :param t: 表示时刻的tensor
        :param tau: LIF神经元的积分时间常数
        :param tau_s: 突触上的电流的衰减时间常数
        :return: t时刻突触后的LIF神经元的电压值
        '''
        # 指数衰减的脉冲输入
        return (tf.exp(-t / tau) - tf.exp(-t / tau_s)) * tf.cast(t >= 0, tf.float32)
    # inspikes: [batch_size, in_features]
    t = np.arange(T, dtype=np.float32) # t = [0, 1, 2, ..., T-1] shape=[T]
    t = np.reshape(t, [1,1,t.shape[0]]) # [1,1,T]
    t = tf.tile(t, [tf.shape(in_spikes)[0], in_feature, 1]) # shape=[batch_size, in_features, T]
    in_spikes = tf.tile(tf.expand_dims(in_spikes, -1), [1,1,T], 'tile0') # shape=[batch_size, in_features, T]
    expand_tau = False # for dagger SDK inference, ture and false get same result
    if not expand_tau:
        in_spikes = v0 * psp_kernel(t - in_spikes, tau, tau_s) * tf.cast(in_spikes >= 0, tf.float32) # shape=[batch_size, in_features, T]
    else:
        tau = tf.expand_dims(tau, -1)
        tau = tf.expand_dims(tau, -1)
        tau = tf.expand_dims(tau, -1)
        tau = tf.tile(tau, [tf.shape(in_spikes)[0], in_feature, T])
        tau_s = tf.expand_dims(tau_s, -1)
        tau_s = tf.expand_dims(tau_s, -1)
        tau_s = tf.expand_dims(tau_s, -1)
        tau_s = tf.tile(tau_s, [tf.shape(in_spikes)[0], in_feature, T])
        tmp = t - in_spikes
        in_spikes = v0 * ( tf.exp(- tf.div(tmp, tau) ) - tf.exp(- tf.div(tmp, tau_s)) ) * tf.cast(tmp >= 0, tf.float32) * tf.cast(in_spikes >= 0, tf.float32)  
    in_spikes = tf.transpose(in_spikes, [0, 2, 1], name='trans0') # shape=[batch_size, T, in_features]
    if ret_type == 'v_max':
        in_spikes = tf.nn.max_pool1d(in_spikes, T, 1, 'VALID', data_format='NWC', name='v_out_global_max_pool') # shape=[batch_size, 1, in_features]
    else:
        raise NotImplementedError('not implemented')
    in_spikes = tf.squeeze(in_spikes, 1) # shape=[batch_size, in_features]
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=None,
                        biases_initializer = None,
                        normalizer_fn=None):
        with tf.variable_scope(scope, 'fcEventNet', [in_spikes], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training): 
                    # v_out = slim.fully_connected(in_spikes, 64, activation_fn=None, scope='fc0') # shape=[batch_size, T, 64]
                    # v_out = slim.fully_connected(v_out, 32, activation_fn=None, scope='fc1') # shape=[batch_size, T, 32]
                    # v_out = slim.fully_connected(v_out, out_feature, activation_fn=None, scope='fc2') # shape=[batch_size, T, out_features]
                    v_out = slim.fully_connected(in_spikes, out_feature, activation_fn=None, scope='fc0') # shape=[batch_size, T, 64]

    v_out = tf.reshape(v_out, [tf.shape(v_out)[0], out_feature], name='logits') # shape=[batch_size, out_features]
    return v_out

def net_PureFC(in_spikes, out_feature, is_training, reuse=None, scope='fcEventNet'):
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=None,
                        biases_initializer = None,
                        normalizer_fn=None):
        with tf.variable_scope(scope, 'fcEventNet', [in_spikes], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training): 
                    v_out = slim.fully_connected(in_spikes, 64, activation_fn=None, scope='fc0') # shape=[batch_size, 64]
                    v_out = slim.fully_connected(v_out, 32, activation_fn=None, scope='fc1') # shape=[batch_size, 32]
                    v_out = slim.fully_connected(v_out, out_feature, activation_fn=None, scope='fc2') # shape=[batch_size, out_features]
        v_out = tf.reshape(v_out, [tf.shape(v_out)[0], out_feature], name='logits')
    return v_out

def fc_event_net_A(input, T, m, n, k, class_num, is_training):
    n, m, mu, sigma2 = init_gaussian_tuning(n = n, m=m, x_min=np.zeros((1), dtype=np.float32), x_max=np.ones((1), dtype=np.float32))
    t_max, v0, tau, tau_s = init_net()
    
    mu = tf.constant(mu)
    sigma2 = tf.constant(sigma2)
    t_max = tf.constant(t_max)
    v0 = tf.constant(v0)

    x = tf.expand_dims(input, 1, name='image_batch_unsq') # [batch_size, n, 784], n=1
    in_spikes = gaussian_encode(x, k, n, m, mu, sigma2, T) # [batch_size, n, k, m]
    in_spikes = tf.reshape(in_spikes, [tf.shape(in_spikes)[0], k*n*m], name='in_spikes_reshape')  # [batch_size, k*n*m]
    out_spikes_counter_tensor = net(in_spikes, k*m, class_num, T, t_max, v0, tau, tau_s, 'v_max', is_training=is_training) # [batch_size, 10]
    return out_spikes_counter_tensor

def pure_fc_net(input, k, class_num, is_training):
    out_spikes_counter_tensor = net_PureFC(input, class_num, is_training)
    return out_spikes_counter_tensor