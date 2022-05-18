# https://github.com/taki0112/Densenet-Tensorflow/blob/master/MNIST/Densenet_MNIST.py
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from models import fc_lif_net_clock_A, fc_lif_net_clock_B

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

model_type = 'B'
T = 10
tau = 2.0
init_learning_rate = 1e-3
epsilon = 1e-8 # AdamOptimizer epsilon
class_num = 10
batch_size = 64
total_epochs = 100

def train():

    batch_images = tf.placeholder(tf.float32, shape=[None, 784], name='image_batch')

    label = tf.placeholder(tf.float32, shape=[None, 10])

    training_flag = tf.placeholder(tf.bool)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    if model_type == 'A':
        out_spikes_counter_tensor = fc_lif_net_clock_A(batch_images, training_flag, T=T, tau=tau, reuse=tf.AUTO_REUSE)
    elif model_type == 'B':
        out_spikes_counter_tensor = fc_lif_net_clock_B(batch_images, training_flag, T=T, tau=tau, reuse=tf.AUTO_REUSE)

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

                if step % 100 == 0:
                    global_step += 100
                    train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                    # accuracy.eval(feed_dict=feed_dict)
                    print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                    writer.add_summary(train_summary, global_step=epoch)

                test_feed_dict = {
                    batch_images: mnist.test.images,
                    label: mnist.test.labels,
                    learning_rate: epoch_learning_rate,
                    training_flag : False
                }
            
            accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
            print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
            # writer.add_summary(test_summary, global_step=epoch)

            test_inference_time = False
            if test_inference_time:
                batch_size_test = 10000
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

        subdir = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
        saver.save(sess=sess, save_path=os.path.join('data/snn_trained_model', 'snn_clock_mnist_'+subdir, 'snn_clock_mnist.ckpt'))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    train()