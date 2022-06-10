import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
import tensorflow.contrib.slim as slim

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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
    x_min_ex = np.tile(np.expand_dims(x_min, axis=-1), [1, m])
    x_max_ex = np.tile(np.expand_dims(x_max, axis=-1), [1, m])
    mu = (2 * i - 3) / 2 * (x_max_ex - x_min_ex) / (m - 2)
    sigma2 = np.power(np.expand_dims(1 / 1.5 * (x_max - x_min) / (m - 2), -1), 2)
    sigma2 = np.tile(sigma2, [1, m])
    return n, m, mu, sigma2


def gaussian_encode(x: np.array,  n, m, mu, sigma2, max_spike_time=50):
    def cond1(x1, x2):
        return x1 >= x2
    '''
    :param x: shape=[batch_size, n, k]，batch_size个数据，每个数据含有n个特征，每个特征中有k个数据
    :param max_spike_time: 最大（最晚）脉冲发放时间，也可以称为编码时间窗口的长度
    :return: out_spikes, shape=[batch_size, n, k, m]，将每个数据编码成了m个神经元的脉冲发放时间
    '''

    x_shape = tf.shape(x, name='x_shape')
    x = tf.transpose(x, [0, 2, 1], name='x_transpose') # [batch_size, n, k] -> [batch_size, k, n]
    x_shape2 = tf.shape(x, name='x_shape1')
    x = tf.reshape(x, [x_shape2[0]*x_shape2[1], x_shape2[2]], name='x_reshape') # shape=[batch_size * k, n]
    x = tf.expand_dims(x, 1, name='x_expand') 
    x = tf.tile(x, [1,1,m], name='x_tile') # shape=[batch_size * k, n, m]
    # 计算x对应的m个高斯函数值
    x = tf.square(x - mu, name='x_square')
    y = tf.exp(- x / 2 / sigma2) # shape=[batch_size * k, n, m]
    # out_spikes = (max_spike_time * (1 - y)).round()
    # out_spikes[out_spikes >= max_spike_time] = -1  # -1表示无脉冲发放
    out_spikes = -1 * tf.cast(y<=0.5/max_spike_time, tf.float32) # use 0.5/max_spike_time to substitude round
    out_spikes += tf.floor(max_spike_time * (1 - y)+0.5, name='spikes_round') * tf.cast(y>0.5/max_spike_time, tf.float32)
 
    out_spikes_shape = tf.shape(out_spikes, name='out_spikes_shape')
    out_spikes = tf.reshape(out_spikes, [x_shape[0], x_shape[2], out_spikes_shape[1], out_spikes_shape[2]], name='out_spikes_reshape') # shape: [batch_size * k, n, m] -> [batch_size, k, n, m]
    out_spikes = tf.transpose(out_spikes, [0,2,1,3], name='out_spikes_transpose') # shape: [batch_size, k, n, m] -> [batch_size, n, k, m]
    return out_spikes

def init_net(tau=15.0, tau_s=15.0 / 4, v_threshold=1.0):
        tau = np.float32(tau)
        tau_s = np.float32(tau_s)
        v_threshold = np.float32(v_threshold)
        t_max = (tau * tau_s * np.log(tau / tau_s)) / (tau - tau_s)
        v0 = v_threshold / (np.exp(-t_max / tau) - np.exp(-t_max / tau_s))
        return t_max, v0, tau, tau_s

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

    t = np.arange(T, dtype=np.float32) # t = [0, 1, 2, ..., T-1] shape=[T]
    t = np.reshape(t, [1,1,t.shape[0]])
    t = tf.tile(t, [tf.shape(in_spikes)[0], in_feature, 1])
    in_spikes = tf.tile(tf.expand_dims(in_spikes, -1), [1,1,T]) # shape=[batch_size, in_features, T]
    in_spikes = v0 * psp_kernel(t - in_spikes, tau, tau_s) * tf.cast(in_spikes >= 0, tf.float32) # shape=[batch_size, in_features, T]
    in_spikes = tf.transpose(in_spikes, [0, 2, 1], name='in_spikes_trans0') # shape=[batch_size, T, in_features]
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=None,
                        biases_initializer = None,
                        normalizer_fn=None):
        with tf.variable_scope(scope, 'fcEventNet', [in_spikes], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
                    
                    v_out = slim.fully_connected(in_spikes, 64, activation_fn=None, scope='fc0')
                    v_out = slim.dropout(v_out, 0.5)
                    v_out = slim.fully_connected(v_out, 32, activation_fn=None, scope='fc1')
                    v_out = slim.dropout(v_out, 0.5)
                    v_out = slim.fully_connected(v_out, out_feature, activation_fn=None, scope='fc2') # shape=[batch_size, T, out_features]
                    v_out = slim.dropout(v_out, 0.5)

    if ret_type == 'v_max':
        v_out = tf.nn.max_pool1d(v_out, 100, 1, 'VALID', data_format='NWC', name='v_out_global_max_pool') # shape=[batch_size, 1, out_features]
    else:
        raise NotImplementedError('not implemented')
    v_out = tf.squeeze(v_out, 1)
    return v_out


def train():

    subdir = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

    T = 100
    m = 16
    n = 1
    k = 28 * 28
    init_learning_rate = 1e-3
    epsilon = 1e-8 # AdamOptimizer epsilon
    class_num = 10
    batch_size = 64
    total_epochs = 100

    batch_images = tf.placeholder(tf.float32, shape=[None, 784], name='image_batch')

    label = tf.placeholder(tf.float32, shape=[None, 10])

    training_flag = tf.placeholder(tf.bool)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    n, m, mu, sigma2 = init_gaussian_tuning(n = n, m=m, x_min=np.zeros((1), dtype=np.float32), x_max=np.ones((1), dtype=np.float32))
    t_max, v0, tau, tau_s = init_net()
    # n =tf.constant(n)
    # m = tf.constant(m)
    mu = tf.constant(mu)
    sigma2 = tf.constant(sigma2)
    t_max = tf.constant(t_max)
    v0 = tf.constant(v0)

    x = tf.expand_dims(batch_images, 1, name='image_batch_unsq')
    in_spikes = gaussian_encode(x, n, m, mu, sigma2, T) # [batch_size, n, k, m]
    in_spikes = tf.reshape(in_spikes, [tf.shape(in_spikes)[0], k*n*m])  # [batch_size, k*n*m]
    out_spikes_counter_tensor = net(in_spikes, k*m, class_num, T, t_max, v0, tau, tau_s, 'v_max', is_training=False)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out_spikes_counter_tensor))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(out_spikes_counter_tensor, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver(tf.global_variables())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    with sess.as_default():
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', sess.graph)

        global_step = 0
        epoch_learning_rate = init_learning_rate
        best_test_acc = 0.0
        for epoch in range(total_epochs):
            if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
                epoch_learning_rate = epoch_learning_rate / 10

            total_batch = int(mnist.train.num_examples / batch_size)

            for step in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                train_feed_dict = {
                    batch_images: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag : True
                }

                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

                # if step % 100 == 0:
                #     global_step += 100
                #     train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                #     # accuracy.eval(feed_dict=feed_dict)
                #     print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                #     writer.add_summary(train_summary, global_step=epoch)

            batch_size_test = 64
            total_batch_test = int(mnist.test.num_examples / batch_size_test)
            test_correct_sum = 0
            test_sum = 0
            for step in range(total_batch_test):
                batch_x, batch_y = mnist.test.next_batch(batch_size_test)

                test_feed_dict = {
                    batch_images: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag : False
                }

                out_spikes_counter = sess.run(out_spikes_counter_tensor, feed_dict=test_feed_dict)
                test_correct_sum += np.sum(np.argmax(out_spikes_counter, axis=1) == np.argmax(batch_y, axis=1))
                test_sum += batch_x.shape[0]
            test_accuracy = test_correct_sum / test_sum
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                saver.save(sess=sess, save_path=os.path.join('data/snn_trained_model', 'snn_event_mnist_'+subdir, 'snn_event_mnist_best.ckpt'))
                print('e %04d, acc=%.4f (best)' % (epoch+1, test_accuracy)) 
            else:
                print('e %04d, acc=%.4f' % (epoch+1, test_accuracy))
                # writer.add_summary(test_summary, global_step=epoch)

            test_inference_time = False
            if test_inference_time:
                batch_size_test = 100
                total_batch_test = int(mnist.test.num_examples / batch_size_test )
                print('start test...')
                t1 = time.time()
                for step in range(total_batch_test):
                    batch_x, batch_y = mnist.test.next_batch(batch_size_test)
                    test_feed_dict = {
                        batch_images: batch_x,
                        label: batch_y,
                        learning_rate: epoch_learning_rate,
                        training_flag : False
                    }
                    _ = sess.run(out_spikes_counter_tensor, feed_dict=test_feed_dict)
                t2 = time.time()
                print('finished test, time=%f ms per batch' % ((t2-t1)*1000.0/total_batch_test))
                exit(0)

            if epoch == 0 or (epoch+1) % 10 == 0:
                saver.save(sess=sess, save_path=os.path.join('data/snn_trained_model', 'snn_event_mnist_'+subdir, 'snn_event_mnist.ckpt'))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()