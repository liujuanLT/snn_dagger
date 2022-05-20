import numpy as np
import os
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar10_img(file_dir, train_output_dir, test_output_dir, appendix='.jpg'):
    if os.path.exists(train_output_dir) == False:
        os.makedirs(train_output_dir)
    if os.path.exists(test_output_dir) == False:
        os.makedirs(test_output_dir)

    # save train datasets
    for i in range(1, 6):
        data_name = file_dir + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')
        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            # 通道顺序为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 要改成不同的形式的文件只需要将文件后缀修改即可
            img_name = os.path.join(train_output_dir, str(data_dict[b'labels'][j]) + str((i) * 10000 + j) + appendix)
            cv2.imwrite(img_name, img)
        print(data_name + ' is done')
    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)
 
    # save test datasets
    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        # 通道顺序为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
        img_name = os.path.join(test_output_dir, str(test_dict[b'labels'][m]) + str(10000 + m) + appendix)
        cv2.imwrite(img_name, img)
    print(test_data_name + ' is done')
    print('Finish transforming to image')

if __name__ == '__main__':
    cache_dataset_dir = os.path.expanduser('~/.keras/datasets/cifar-10-batches-py')
    train_output_dir = './data/datasets/cifar10/imgs/train'
    test_output_dir =  './data/datasets/cifar10/imgs/test'
    cifar10_img(cache_dataset_dir, train_output_dir, test_output_dir, appendix='.png')