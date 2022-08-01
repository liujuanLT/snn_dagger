import tensorflow
import numpy as np
import cv2
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
test_images, test_labels = mnist.test.next_batch(mnist.test.num_examples)

save_dir = 'data/datasets/MNIST/imgs_tf'
n_map = {}
for i in range(test_images.shape[0]):
    img = test_images[i, ...]
    label = test_labels[i]
    n_map[label] = n_map.get(label, 0)
    n_map[label] += 1
    filename = os.path.join(save_dir, str(label), str(n_map[label])+'.png')
    img = np.reshape(img, (28, 28, 1))
    ret = cv2.imwrite(filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) # no compression
print('done')