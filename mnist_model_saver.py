import tensorflow as tf
import numpy as np
import os
from models import fc_lif_net_clock_A, fc_lif_net_clock_B, fc_lif_net_clock_C, fc_lif_net_clock_A_Nlayers
from models_event import fc_event_net_A, pure_fc_net
from modelsaver import TFDaggerAdapter, visual_lt
ltsdk_version=None

def ignore_nodes_filter():
    from lt_sdk.proto import node_filters
    keep_nodes = node_filters.or_filter(
        node_filters.name_starts_with_filter("reshape_pre0"),
        node_filters.name_starts_with_filter("reshape_pre1"),
        node_filters.name_starts_with_filter("reshape0"),
        node_filters.name_starts_with_filter("reshape1")
    )
    return node_filters.not_filter(keep_nodes)
    # return node_filters.and_filter(keep_nodes)

class MnistModelSaver(TFDaggerAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mnist_acc_test(self, batch_size_test=1, resfile='data/mnist_acc_test.txt'):
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

        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        total_batch_test = int(mnist.test.num_examples / batch_size_test )
        print('start test...')
        correct_num = 0
        sample_num = 0
        fid = open(resfile, 'w')
        for step in range(total_batch_test):
            batch_x, batch_y = mnist.test.next_batch(batch_size_test)
            infer_res = sess.run(output_tensor, feed_dict={input_tensor: batch_x})
            correct_num += np.sum(np.argmax(infer_res, 1) == np.argmax(batch_y, 1))
            sample_num += infer_res.shape[0]
            if step % 20 == 0:
                acc = correct_num / sample_num
                print(f'acc={acc}')
                fid.write('step = %d, acc=%f\n' % (step, acc))
                fid.flush()
        fid.close()
        acc = correct_num / sample_num
        return acc


    def lt_mnist_acc_test(self, lt_graph_path, input_type='TFGraphDef', resfile='data/lt_mnist_acc_test.txt'):
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
        # config.sw_config.ignore_nodes_filter.CopyFrom(ignore_nodes_filter().as_proto())
        # print("sw_config *** = ", config.sw_config)
        graph = lt.import_graph(lt_graph_path, config, graph_types_pb2.LGFProtobuf)

        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        batch_size_test=1
        total_batch_test = int(mnist.test.num_examples / batch_size_test )
        correct_num = 0
        sample_num = 0
        fid = open(resfile, 'w')
        for step in range(total_batch_test):
            batch_x, batch_y = mnist.test.next_batch(batch_size_test)
            named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [batch_x])
            batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size_test)
            infer_res = infer_process(graph, batch_input, config)
            correct_num += np.sum(np.argmax(infer_res, 1) == np.argmax(batch_y, 1))
            sample_num += infer_res.shape[0]
            if step % 20 == 0:
                acc = correct_num / sample_num
                print(f'acc={acc}')
                fid.write('step = %d, acc=%f\n' % (step, acc))
                fid.flush()

        fid.close()
        acc = correct_num / sample_num
        return acc

class SnnClockMnistModelSaverA(MnistModelSaver):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'TF')
        self.net = fc_lif_net_clock_A

    def address_input_output_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_name)
        out_spikes_counter_tensor = self.net(self.input_placeholder, is_training=False, T=10, tau=2., reuse=tf.AUTO_REUSE)
        self.embeddings_placeholder = tf.add(0.0, out_spikes_counter_tensor, name=self.output_name)

    def preprocess_single_sample(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            try:
                raw_image = tf.io.read_file(image_data_or_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_data_or_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=1)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 784])
        else:
            image_data = tf.reshape(image_data_or_path, [1, 784])
        return image_data

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')

class SnnClockMnistModelSaverB(MnistModelSaver):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'TF')
        self.net = fc_lif_net_clock_B


    def address_input_output_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_name)
        out_spikes_counter_tensor = self.net(self.input_placeholder, is_training=False, T=10, tau=2., reuse=tf.AUTO_REUSE)
        self.embeddings_placeholder = tf.add(0.0, out_spikes_counter_tensor, name=self.output_name)

    def preprocess_single_sample(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            try:
                raw_image = tf.io.read_file(image_data_or_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_data_or_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=1)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 784])
        else:
            image_data = tf.reshape(image_data_or_path, [1, 784])
        return image_data

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')
            

class SnnClockMnistModelSaverC(MnistModelSaver):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'TF')
        self.net = fc_lif_net_clock_C


    def address_input_output_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_name)
        out_spikes_counter_tensor = self.net(self.input_placeholder, is_training=False, T=10, tau=2., reuse=tf.AUTO_REUSE)
        self.embeddings_placeholder = tf.add(0.0, out_spikes_counter_tensor, name=self.output_name)

    def preprocess_single_sample(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            try:
                raw_image = tf.io.read_file(image_data_or_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_data_or_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=1)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 784])
        else:
            image_data = tf.reshape(image_data_or_path, [1, 784])
        return image_data

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')


class SnnClockMnistModelSaverANlayer(MnistModelSaver):
    def __init__(self, image_shape, nlayers=2):
        super().__init__(image_shape, 'image_batch', 'logits', 'TF')
        self.net = fc_lif_net_clock_A_Nlayers
        self.nlayers = nlayers

    def address_input_output_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_name)
        out_spikes_counter_tensor = self.net(self.input_placeholder, is_training=False, T=10, tau=2., nlayers=self.nlayers, reuse=tf.AUTO_REUSE)
        self.embeddings_placeholder = tf.add(0.0, out_spikes_counter_tensor, name=self.output_name)

    def preprocess_single_sample(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            try:
                raw_image = tf.io.read_file(image_data_or_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_data_or_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=1)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 784])
        else:
            image_data = tf.reshape(image_data_or_path, [1, 784])
        return image_data

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')



class SnnEventMnistModelSaverA(MnistModelSaver):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'TF')
       

    def address_input_output_placeholder(self):
        T = 100
        m = 16
        n = 1
        k = 28 * 28
        class_num = 10
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, k], name=self.input_name)
        self.embeddings_placeholder = fc_event_net_A(self.input_placeholder, T, m, n, k, class_num, False)

    def preprocess_single_sample(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            try:
                raw_image = tf.io.read_file(image_data_or_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_data_or_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=1)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 784])
        else:
            image_data = tf.reshape(image_data_or_path, [1, 784])
        return image_data

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')


class SnnEventMnistModelSaverPureFC(MnistModelSaver):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'TF')
       
    def address_input_output_placeholder(self):
        k = 28 * 28
        class_num = 10
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, k], name=self.input_name)
        self.embeddings_placeholder = pure_fc_net(self.input_placeholder, k, class_num, False)
    
    def preprocess_single_sample(self, image_data_or_path):
        if isinstance(image_data_or_path, str):
            try:
                raw_image = tf.io.read_file(image_data_or_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_data_or_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=1)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 784])
        else:
            image_data = tf.reshape(image_data_or_path, [1, 784])
        return image_data

    def _adapt_to_dagger(self):
            if not self.restored:
                raise RuntimeError('model must been restored')


def test_snn_clock_mnist_A():
    modelsaver = SnnClockMnistModelSaverA((28,28))
    model_dir_path = 'data/snn_trained_model/snn_clock_mnist_A_2022-05-18-00-34'
    lt_model_dir_path = os.path.join(model_dir_path, 'ltsdk'+ltsdk_version)
    ckpt_file = os.path.join(model_dir_path, 'snn_clock_mnist.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_clock_mnist.pb')
    out_pb_file_adaptlt = os.path.join(lt_model_dir_path, 'snn_clock_mnist_adaptlt.pb')
    image_path = 'data/datasets/MNIST/imgs/test/3/30.png'
    modelsaver.load_ckpt(ckpt_file)
    modelsaver.save_to_saved_model(saved_model_dir, need_adapt_to_dagger=True, force_rewrite=True)
    modelsaver.save_to_pb(out_pb_file)
    modelsaver.save_to_pb(out_pb_file_adaptlt)
    modelsaver.load_pb_graphdef(out_pb_file_adaptlt)
    embed_res = modelsaver.inference(image_path, print_info=True)
    cls = np.argmax(embed_res, axis=1)[0]
    print(f'class={cls}')
    acc = modelsaver.mnist_acc_test(resfile=os.path.join(model_dir_path, 'mnist_acc_test.txt'))
    print(f'acc={acc}')
    ltgraph_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel', calib_data='mnist', calib_sample_num=500)
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(lt_model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')

            
def test_snn_clock_mnist_B():
    modelsaver = SnnClockMnistModelSaverB((28,28))
    model_dir_path = 'data/snn_trained_model/snn_clock_mnist_B_2022-05-18-12-15'
    lt_model_dir_path = os.path.join(model_dir_path, 'ltsdk'+ltsdk_version)
    ckpt_file = os.path.join(model_dir_path, 'snn_clock_mnist.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_clock_mnist.pb')
    out_pb_file_adaptlt = os.path.join(lt_model_dir_path, 'snn_clock_mnist_adaptlt.pb')
    image_path = 'data/datasets/MNIST/imgs/test/3/30.png'
    modelsaver.load_ckpt(ckpt_file)
    modelsaver.save_to_saved_model(saved_model_dir, need_adapt_to_dagger=True, force_rewrite=True)
    modelsaver.save_to_pb(out_pb_file)
    modelsaver.save_to_pb(out_pb_file_adaptlt)
    modelsaver.load_pb_graphdef(out_pb_file_adaptlt)
    embed_res = modelsaver.inference(image_path, print_info=True)
    cls = np.argmax(embed_res, axis=1)[0]
    print(f'class={cls}')
    acc = modelsaver.mnist_acc_test(resfile=os.path.join(model_dir_path, 'mnist_acc_test.txt'))
    print(f'acc={acc}')
    ltgraph_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel', calib_data='mnist', calib_sample_num=500)
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(lt_model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')

      
def test_snn_clock_mnist_C():
    modelsaver = SnnClockMnistModelSaverC((28,28))
    model_dir_path = 'data/snn_trained_model/snn_clock_mnist_C_2022-05-31-12-01'
    lt_model_dir_path = os.path.join(model_dir_path, 'ltsdk'+ltsdk_version)
    ckpt_file = os.path.join(model_dir_path, 'snn_clock_mnist.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_clock_mnist.pb')
    out_pb_file_adaptlt = os.path.join(lt_model_dir_path, 'snn_clock_mnist_adaptlt.pb')
    image_path = 'data/datasets/MNIST/imgs/test/3/30.png'
    modelsaver.load_ckpt(ckpt_file)
    modelsaver.save_to_saved_model(saved_model_dir, need_adapt_to_dagger=True, force_rewrite=True)
    modelsaver.save_to_pb(out_pb_file)
    modelsaver.save_to_pb(out_pb_file_adaptlt)
    modelsaver.load_pb_graphdef(out_pb_file_adaptlt)
    embed_res = modelsaver.inference(image_path, print_info=True)
    cls = np.argmax(embed_res, axis=1)[0]
    print(f'class={cls}')
    acc = modelsaver.mnist_acc_test(resfile=os.path.join(model_dir_path, 'mnist_acc_test.txt'))
    print(f'acc={acc}')
    ltgraph_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel', calib_data='mnist', calib_sample_num=500)
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(lt_model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')


def test_snn_clock_mnist_A_Nlayers():
    modelsaver = SnnClockMnistModelSaverANlayer((28,28), nlayers=20)
    # model_dir_path = 'data/snn_trained_model/snn_clock_mnist-A_4layers_2022-06-02-16-33'
    model_dir_path = 'data/snn_trained_model/snn_clock_mnist-A_20layers_2022-06-02-14-52'
    lt_model_dir_path = os.path.join(model_dir_path, 'ltsdk'+ltsdk_version)
    ckpt_file = os.path.join(model_dir_path, 'snn_clock_mnist.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_clock_mnist.pb')
    out_pb_file_adaptlt = os.path.join(lt_model_dir_path, 'snn_clock_mnist_adaptlt.pb')
    image_path = 'data/datasets/MNIST/imgs/test/3/30.png'
    modelsaver.load_ckpt(ckpt_file)
    modelsaver.save_to_saved_model(saved_model_dir, need_adapt_to_dagger=True, force_rewrite=True)
    modelsaver.save_to_pb(out_pb_file)
    modelsaver.save_to_pb(out_pb_file_adaptlt)
    modelsaver.load_pb_graphdef(out_pb_file_adaptlt)
    embed_res = modelsaver.inference(image_path, print_info=True)
    cls = np.argmax(embed_res, axis=1)[0]
    print(f'class={cls}')
    # acc = modelsaver.mnist_acc_test(resfile=os.path.join(model_dir_path, 'mnist_acc_test.txt'))
    # print(f'acc={acc}')
    ltgraph_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel', calib_data='mnist', calib_sample_num=500)
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(lt_model_dir_path, 'snn_clock_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(lt_model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')


def test_snn_event_mnist_A():
    modelsaver = SnnEventMnistModelSaverA((28,28))
    model_dir_path = 'data/snn_trained_model/snn_event_mnist_2022-06-23-11-01'
    lt_model_dir_path = os.path.join(model_dir_path, 'ltsdk'+ltsdk_version)
    if not os.path.exists(lt_model_dir_path):
        os.makedirs(lt_model_dir_path)
    ckpt_file = os.path.join(model_dir_path, 'snn_event_mnist_best.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_event_mnist.pb')
    out_pb_file_adaptlt = os.path.join(lt_model_dir_path, 'snn_event_mnist_adaptlt.pb')
    image_path = 'data/datasets/MNIST/imgs/test/3/30.png'
    # image_path = 'data/datasets/MNIST/imgs/test/5/7498.png'
    modelsaver.load_ckpt(ckpt_file)
    modelsaver.save_to_saved_model(saved_model_dir, need_adapt_to_dagger=True, force_rewrite=True)
    modelsaver.save_to_pb(out_pb_file)
    modelsaver.save_to_pb(out_pb_file_adaptlt)
    modelsaver.load_pb_graphdef(out_pb_file_adaptlt)
    embed_res = modelsaver.inference(image_path, print_info=True)
    cls = np.argmax(embed_res, axis=1)[0]
    print(f'class={cls}')
    acc = modelsaver.mnist_acc_test(resfile=os.path.join(model_dir_path, 'mnist_acc_test.txt'))
    print(f'acc={acc}')
    ltgraph_file = os.path.join(lt_model_dir_path, 'snn_event_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel', calib_data='mnist', calib_sample_num=500)
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(lt_model_dir_path, 'snn_event_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(lt_model_dir_path, 'snn_event_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(lt_model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')


def test_snn_event_mnist_PureFC():
    modelsaver = SnnEventMnistModelSaverPureFC((28,28))
    model_dir_path = 'data/snn_trained_model/snn_event_mnist_2022-06-23-11-11'
    lt_model_dir_path = os.path.join(model_dir_path, 'ltsdk'+ltsdk_version)
    if not os.path.exists(lt_model_dir_path):
        os.makedirs(lt_model_dir_path)
    ckpt_file = os.path.join(model_dir_path, 'snn_event_mnist_best.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_event_mnist.pb')
    out_pb_file_adaptlt = os.path.join(lt_model_dir_path, 'snn_event_mnist_adaptlt.pb')
    image_path = 'data/datasets/MNIST/imgs/test/3/30.png'
    # image_path = 'data/datasets/MNIST/imgs/test/5/7498.png'
    modelsaver.load_ckpt(ckpt_file)
    modelsaver.save_to_saved_model(saved_model_dir, need_adapt_to_dagger=True, force_rewrite=True)
    modelsaver.save_to_pb(out_pb_file)
    modelsaver.save_to_pb(out_pb_file_adaptlt)
    modelsaver.load_pb_graphdef(out_pb_file_adaptlt)
    embed_res = modelsaver.inference(image_path, print_info=True)
    cls = np.argmax(embed_res, axis=1)[0]
    print(f'class={cls}')
    acc = modelsaver.mnist_acc_test(resfile=os.path.join(model_dir_path, 'mnist_acc_test.txt'))
    print(f'acc={acc}')
    ltgraph_file = os.path.join(lt_model_dir_path, 'snn_event_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel', calib_data='mnist', calib_sample_num=500)
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(lt_model_dir_path, 'snn_event_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(lt_model_dir_path, 'snn_event_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(lt_model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ltsdk_version = '0.0.1'
    test_snn_clock_mnist_A()
    test_snn_clock_mnist_B()
    test_snn_clock_mnist_C()
    test_snn_clock_mnist_A_Nlayers()
    test_snn_event_mnist_A()
    test_snn_event_mnist_PureFC()