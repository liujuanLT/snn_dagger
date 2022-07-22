# https://github.com/taki0112/Densenet-Tensorflow/blob/master/MNIST/Densenet_MNIST.py
import os
import tensorflow as tf
import numpy as np
import time
from models import fc_lif_net_clock_A, fc_lif_net_clock_B

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:, :, :, :] / 255.
x_test = x_test[:, :, :, :] / 255.
y_train = np.squeeze(y_train, axis=1)
y_test = np.squeeze(y_test, axis=1)

model_type = 'A'
T = 10
tau = 2.0
init_learning_rate = 1e-3
epsilon = 1e-8 # AdamOptimizer epsilon
class_num = 10
batch_size = 64
total_epochs = 50

test_batch_size = 128

class NpDataloader(object):
    def __init__(self, data, labels, batch_size, shuffle, drop_last):
        # if flat_data:
        #     self.data = np.
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.ibatch = 0
        self.nsample = data.shape[0]
        self.nbatch = (self.nsample - 1) // self.batch_size + 1
        # TODO: shuffle
        self.drop_last = drop_last

    def get_total_batch(self):
        return self.nbatch

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

def train():
    subdir = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

    train_data_loader = NpDataloader(x_train, y_train, batch_size, None, True)
    test_data_loader = NpDataloader(x_test, y_test, test_batch_size, None, True)

    batch_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='image_batch')
    batch_images_flatten = tf.reshape(batch_images, [-1, 32*32*3], name='flatten_input')

    label = tf.placeholder(tf.uint8, shape=[None])
    label_onehot = tf.one_hot(label, 10) # after one_hot, datatepe turns to float32

    training_flag = tf.placeholder(tf.bool)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    if model_type == 'A':
        out_spikes_counter_tensor = fc_lif_net_clock_A(batch_images_flatten, training_flag, T=T, tau=tau, reuse=tf.AUTO_REUSE)
    elif model_type == 'B':
        out_spikes_counter_tensor = fc_lif_net_clock_B(batch_images_flatten, training_flag, T=T, tau=tau, reuse=tf.AUTO_REUSE)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot, logits=out_spikes_counter_tensor))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(out_spikes_counter_tensor, 1), tf.argmax(label_onehot, 1))
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
        for epoch in range(total_epochs):
            if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
                epoch_learning_rate = epoch_learning_rate / 10
            step = 0
            for batch_x, batch_y in train_data_loader:
                train_feed_dict = {
                    batch_images: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag : True
                }

                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

                if step % 100 == 0:
                    global_step += 100
                    train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                    # accuracy.eval(feed_dict=feed_dict)
                    print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                    writer.add_summary(train_summary, global_step=epoch)

                step += 1
            
            for test_batch_x, test_batch_y in test_data_loader:
                    test_feed_dict = {
                        batch_images: test_batch_x,
                        label: test_batch_y,
                        learning_rate: epoch_learning_rate,
                        training_flag : False
                    }

            accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
            print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
            # writer.add_summary(test_summary, global_step=epoch)

            saver.save(sess=sess, save_path=os.path.join('data/snn_trained_model', 'snn_clock_cifar10_'+model_type+'_'+subdir, 'snn_clock_cifar10.ckpt'))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train()