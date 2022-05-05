import os,sys
import argparse
import numpy as np
import struct
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
# from spikingjelly_tf.clock_driven import neuron, encoding, functional

         

class NpDataloader(object):
    def __init__(self, data, labels, batch_size, shuffle, drop_last):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.ibatch = 0
        self.nsample = data.shape[0]
        self.nbatch = (self.nsample - 1) // self.batch_size + 1
        # TODO: shuffle
        self.drop_last = drop_last

    def __iter__(self):
        return self

    def __next__(self):
        if self.drop_last:
            if (self.ibatch+1)*self.batch_size <= self.nsample:
                batch_data = self.data[self.ibatch * self.batch_size: (self.ibatch+1) * self.batch_size]
                batch_labels = self.labels[self.ibatch * self.batch_size: (self.ibatch+1) * self.batch_size]
                self.ibatch += 1
            else:
                self.ibatch = 0
                raise StopIteration
        else:
            if self.ibatch < self.nbatch:
                batch_data = self.data[self.ibatch * self.batch_size: min(self.nsample, (self.ibatch+1) * self.batch_size)]
                batch_labels = self.labels[self.ibatch * self.batch_size: min(self.nsample, (self.ibatch+1) * self.batch_size)]
                self.ibatch += 1
            else:
                self.ibatch = 0
                raise StopIteration
        return batch_data, batch_labels

def load_mnist(mnist_dir, data_type='train'):
    type_flag = 'train' if data_type=='train' else 't10k'
    labels_path = os.path.join(mnist_dir, '%s-labels-idx1-ubyte' % type_flag)
    imgs_path = os.path.join(mnist_dir, '%s-images-idx3-ubyte' % type_flag)
    with open(labels_path, 'rb') as fid:
        magic, n = struct.unpack('>II', fid.read(8))
        labels = np.fromfile(fid, dtype=np.uint8)
    with open(imgs_path, 'rb') as fid:
        magic, n, rows, cols = struct.unpack('>IIII', fid.read(16))
        assert rows == 28 and cols==28
        images = np.fromfile(fid, dtype=np.uint8).reshape(len(labels), 1, rows, cols)
    images = (images / 255.0).astype(np.float32)
    labels = labels.astype(np.int64)
    return images, labels

def test1():
    samples = np.random.rand(100,4,5)
    labels = np.random.rand(100,1)
    loader = NpDataloader(samples, labels, 32, True, True)
    for data, label in loader:
        print(data.shape)
        print(label.shape)

def test2():
    samples, labels = load_mnist('/home/jliu/codes/MNIST/raw', 'test')
    loader = NpDataloader(samples, labels, 32, True, True)
    for data, label in loader:
        print(data.shape)
        print(label.shape)

import matplotlib.pyplot as plt

def test3():
    samples, labels = load_mnist('/home/jliu/codes/MNIST/raw', 'train')
    fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex=True, sharey = True)
    ax = ax.flatten()
    for i in range(10):
        img = samples[labels == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.savefig('data/1.png')
    print('data/1.png')

def LIFNode(inputs, lif_state = None, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function = tf.sigmoid,
                 detach_reset: bool = False):

    assert isinstance(v_reset, float) or v_reset is None
    assert isinstance(v_threshold, float)
    assert isinstance(detach_reset, bool)

    def create_init_v():
        if v_reset is None:
            v_ = 0.
        else:
            v_ = v_reset
        return v_

    def neuronal_charge(x, v_):
        if decay_input:
            if v_reset is None or v_reset == 0.:
                v_ = v_ + (x - v_) / tau
            else:
                v_ = v_ + (x - (v_ - v_reset)) / tau
        else:
            if v_reset is None or v_reset == 0.:
                v_ = v_ * (1. - 1. / tau) + x
            else:
                v_ = v_ - (v_ - v_reset) / tau + x
        return v_

    def neuronal_fire(v_):
        return surrogate_function(v_ - v_threshold)

    def neuronal_reset(spike, v_):
        spike_d = spike
        if v_reset is None:
            # soft reset
            v_ = v_ - spike_d * v_threshold
        else:
            # hard reset
            v_ = (1. - spike_d) * v_ + spike_d * v_reset
        return v_

    if lif_state is not None:
        v = lif_state
    else:
        v = create_init_v()
    v = neuronal_charge(inputs, v)
    spike = neuronal_fire(v)
    new_v = neuronal_reset(spike, v)
    return spike, new_v

def fc_lif_net(inputs, lif_state, is_training, tau, weight_decay=0., reuse=None, scope='fcLif'):
    # TODO: arg_scope, reuse?
    batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
    
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope(scope, 'fcLif', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
                    net = slim.flatten(inputs, scope='flatten0')
                    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc0')
                    # num_spikes, new_lif_state = neuron.LIFNode(tau=tau)(net, lif_state)
                    num_spikes, new_lif_state = LIFNode(net, lif_state=lif_state, tau=tau)

    return num_spikes, new_lif_state
    # return net, 0


def main(args):

    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")

    dataset_dir = args.dataset_dir
    temp_dataset_dir = args.temp_dataset_dir
    log_dir = args.log_dir
    model_dir = args.model_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    tau = args.tau
    epochs = args.epochs

    try:
        train_samples = np.load(os.path.join(temp_dataset_dir, 'train_samples.npy'))
        train_labels = np.load(os.path.join(temp_dataset_dir, 'train_labels.npy'))
        test_samples = np.load(os.path.join(temp_dataset_dir, 'test_samples.npy'))
        test_labels = np.load(os.path.join(temp_dataset_dir, 'test_labels.npy'))
    except Exception:
        train_samples, train_labels = load_mnist(dataset_dir, 'train')
        test_samples, test_labels = load_mnist(dataset_dir, 'test')
        np.save(os.path.join(temp_dataset_dir, 'train_samples.npy'), train_samples)
        np.save(os.path.join(temp_dataset_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(temp_dataset_dir, 'test_samples.npy'), test_samples)
        np.save(os.path.join(temp_dataset_dir, 'test_labels.npy'), test_labels)
    train_data_loader = NpDataloader(train_samples, train_labels, batch_size, None, True)
    test_data_loader = NpDataloader(test_samples, test_labels, batch_size, None, True)

    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1, 28, 28), name='inputs')
    labels_placeholder = tf.placeholder(tf.int64, name='labels')
    phase_placeholder = tf.placeholder(tf.bool, name='phase')
    
    # encoded_inputs = encoding.PoissonEncoder(inputs_placeholder)
    # lessthan = (np.random.rand(*inputs_placeholder.shape) < inputs_placeholder)
    lessthan = (tf.random_uniform(inputs_placeholder.shape) < inputs_placeholder)
    encoded_inputs = tf.cast(lessthan, tf.float32, name='encodes_inputs')
    for t in range(T):
        if t == 0:
            num_spikes_tensor, new_lif_state_tensor = fc_lif_net(encoded_inputs, lif_state=None, tau=tau, is_training=phase_placeholder) # TODO, to float
            out_spikes_counter_tensor = num_spikes_tensor
        else:
            num_spikes_tensor, new_lif_state_tensor = fc_lif_net(encoded_inputs, lif_state=new_lif_state_tensor, tau=tau, is_training=phase_placeholder) # TODO, to float
            out_spikes_counter_tensor += num_spikes_tensor

    out_spikes_counter_frequency_tensor = out_spikes_counter_tensor / T

    label_one_hot = tf.one_hot(labels_placeholder, 10) # TODO, to float
    loss = 2.0 / (28*28) * tf.nn.l2_loss(out_spikes_counter_frequency_tensor - label_one_hot) # TODO, change magic number

    # moving_average_decay = 0.9999
    # update_gradient_vars = tf.global_variables()
    # global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    # inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    opt = tf.train.AdamOptimizer(lr, epsilon=0.1)
    # grads = opt.compute_gradients(loss)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_op = opt.apply_gradients(grads, global_step=global_step)
    train_op = opt.minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    # coord = tf.train.Coordinator()
    # tf.train.start_queue_runners(coord=coord, sess=sess)

    with sess.as_default():
        train_times = 0
        max_test_accuracy = 0
        test_accs = []
        train_accs = []
        for epoch in range(epochs):
            # print("Epoch {}:".format(epoch))
            # print("Training...")
            train_correct_sum = 0
            train_sum = 0
            for img, label in train_data_loader:
                feed_dict = {inputs_placeholder: img, labels_placeholder: label, phase_placeholder: True} # TODO, get intial LIF state
                out_spikes_counter_frequency, loss_, _  = sess.run([out_spikes_counter_frequency_tensor, loss, train_op], feed_dict=feed_dict)
            
                # train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()
                train_correct_sum += np.sum(np.argmax(out_spikes_counter_frequency, axis=1) == label)
                train_sum += label.shape[0]
                train_batch_accuracy = np.mean(np.argmax(out_spikes_counter_frequency, axis=1) == label)
                train_accs.append(train_batch_accuracy)

                train_times += 1
            train_accuracy = train_correct_sum / train_sum

            # evaluate
            # print("Testing...")
            test_correct_sum = 0
            test_sum = 0
            for img, label in test_data_loader:
                feed_dict = {inputs_placeholder: img, labels_placeholder: label, phase_placeholder: False} # TODO, get intial LIF state
                out_spikes_counter, = sess.run([out_spikes_counter_tensor], feed_dict=feed_dict)
                test_correct_sum += np.sum(np.argmax(out_spikes_counter, axis=1) == label)
                test_sum += label.shape[0]
            test_accuracy = test_correct_sum / test_sum
            test_accs.append(test_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)

            print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
            # print()

            # TODO, save model

def parse_args(argv):
    parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

    parser.add_argument('--dataset-dir', default='data/datasets/MNIST/raw', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
    parser.add_argument('--temp-dataset-dir', default='data/temp')
    parser.add_argument('--log-dir', default='data/log_dir', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
    parser.add_argument('--model-dir', default='data/models_dir', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
    parser.add_argument('-T', '--timesteps', default=1, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
    parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
    parser.add_argument('-epochs', '--epochs', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main(parse_args(sys.argv[1:]))
    # test3()


