import tensorflow as tf
import numpy as np
import os
from models import fc_lif_net_clock_A, fc_lif_net_clock_B
from modelsaver import TFDaggerAdapter, visual_lt

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

class Cifar10ModelSaver(TFDaggerAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cifar10_acc_test(self, batch_size_test=1, resfile='data/cifar10_acc_test.txt'):
        if not self.restored:
            raise RuntimeError('model must been restored')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        try:
            # if import pb graphdef then if include import
            input_tensor = tf.get_default_graph().get_tensor_by_name('import/' + self.input_name + ':0')
            output_tensor = tf.get_default_graph().get_tensor_by_name('import/' + self.output_name + ':0')
        except Exception as e:
            # if import ckpt then include no import
            # TODO, actually, inference from restored ckpt is not OK by now
            input_tensor = tf.get_default_graph().get_tensor_by_name(self.input_name + ':0')
            output_tensor = tf.get_default_graph().get_tensor_by_name(self.output_name + ':0')    

        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train[:, :, :, :] / 255.
        x_test = x_test[:, :, :, :] / 255.
        y_train = np.squeeze(y_train, axis=1)
        y_test = np.squeeze(y_test, axis=1)

        test_data_loader = NpDataloader(x_test, y_test, batch_size_test, None, True)

        print('start test...')
        correct_num = 0
        sample_num = 0
        fid = open(resfile, 'w')
        step = 0
        for batch_x, batch_y in test_data_loader:
            infer_res = sess.run(output_tensor, feed_dict={input_tensor: batch_x})
            correct_num += np.sum(np.argmax(infer_res, 1) == batch_y[0])
            sample_num += infer_res.shape[0]
            if step % 20 == 0:
                acc = correct_num / sample_num
                print(f'acc={acc}')
                fid.write('step = %d, acc=%f\n' % (step, acc))
                fid.flush()
            step += 1
        fid.close()
        acc = correct_num / sample_num
        return acc



    def lt_cifar10_acc_test(self, lt_graph_path, input_type='TFGraphDef', resfile='data/lt_cifar10_acc_test.txt'):
        import lt_sdk as lt
        from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
        from lt_sdk.graph.transform_graph import utils as lt_utils

        def infer_process(light_graph=None, calibration_data=None, config=None):
            outputs = lt.run_functional_simulation(light_graph, calibration_data, config)
            for inf_out in outputs.batches:
                for named_ten in inf_out.results:
                    if named_ten.edge_info.name.startswith(self.output_name):   #输出节点名0
                        embed_res = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                        try:
                            embed_res = embed_res.numpy()
                        except Exception:
                            with tf.Session() as sess:
                                embed_res = sess.run(embed_res) # tensor to list
                        return embed_res
            return None

        config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=eval('graph_types_pb2.'+input_type))
        graph = lt.import_graph(lt_graph_path, config)

        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train[:, :, :, :] / 255.
        x_test = x_test[:, :, :, :] / 255.
        y_train = np.squeeze(y_train, axis=1)
        y_test = np.squeeze(y_test, axis=1)

        batch_size_test = 1
        test_data_loader = NpDataloader(x_test, y_test, batch_size_test, None, True)

        print('start test...')
        correct_num = 0
        sample_num = 0
        fid = open(resfile, 'w')
        step = 0
        for batch_x, batch_y in test_data_loader:
            named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [batch_x])
            batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size_test)
            infer_res = infer_process(graph, batch_input, config)
            correct_num += np.sum(np.argmax(infer_res, 1) == batch_y[0])
            sample_num += infer_res.shape[0]
            if step % 20 == 0:
                acc = correct_num / sample_num
                print(f'acc={acc}')
                fid.write('step = %d, acc=%f\n' % (step, acc))
                fid.flush()
            step += 1
        fid.close()
        acc = correct_num / sample_num
        return acc



class SnnClockCifar10ModelSaverA(Cifar10ModelSaver):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'TF')
        self.net = fc_lif_net_clock_A

    def address_input_output_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name=self.input_name)
        batch_images_flatten = tf.reshape(self.input_placeholder, [-1, 32*32*3], name='flatten_input')
        out_spikes_counter_tensor = self.net(batch_images_flatten, is_training=False, T=10, tau=2., reuse=tf.AUTO_REUSE)
        self.embeddings_placeholder = tf.add(0.0, out_spikes_counter_tensor, name=self.output_name)

    def preprocess_single_sample(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            try:
                raw_image = tf.io.read_file(image_data_or_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_data_or_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=3)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 32, 32, 3])
        else:
            image_data = tf.reshape(image_data_or_path, [1, 32, 32, 3])
        return image_data

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')

     
def test_snn_clock_cifar10_A():
    modelsaver = SnnClockCifar10ModelSaverA((32,32,3))
    model_dir_path = 'data/snn_trained_model/snn_clock_cifar10_2022-05-20-15-53'
    ckpt_file = os.path.join(model_dir_path, 'snn_clock_cifar10.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_clock_cifar10.pb')
    out_pb_file_adaptlt = os.path.join(model_dir_path, 'snn_clock_cifar10_adaptlt.pb')
    image_path = 'data/datasets/cifar10/imgs/test/010044.png'
    modelsaver.load_ckpt(ckpt_file)
    modelsaver.save_to_saved_model(saved_model_dir, need_adapt_to_dagger=True, force_rewrite=True)
    modelsaver.save_to_pb(out_pb_file)
    modelsaver.save_to_pb(out_pb_file_adaptlt)
    modelsaver.load_pb_graphdef(out_pb_file_adaptlt)
    embed_res = modelsaver.inference(image_path, print_info=True)
    cls = np.argmax(embed_res, axis=1)[0]
    print(f'class={cls}')
    acc = modelsaver.cifar10_acc_test(resfile=os.path.join(model_dir_path, 'cifar10_acc_test.txt'))
    print(f'acc={acc}')
    ltgraph_file = os.path.join(model_dir_path, 'snn_clock_cifar10_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel')
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(model_dir_path, 'snn_clock_cifar10_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(model_dir_path, 'snn_clock_cifar10_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_cifar10_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(model_dir_path, 'cifar10_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')

if __name__ == '__main__':
    # debug code
    np.random.seed(0)
    tf.set_random_seed(2)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    test_snn_clock_cifar10_A()