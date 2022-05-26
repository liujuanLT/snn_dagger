import tensorflow as tf
from abc import ABC, abstractclassmethod, abstractmethod
from six.moves import xrange  # @UnresolvedImport
from tensorflow.python.framework import graph_util
import numpy as np
import os


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

def visual_lt(lt_graph_path, lt_graph_json_file):
    import json
    from lt_sdk.visuals import lgf_proto_to_json
    json_object = lgf_proto_to_json.main(
        lt_graph_path,
        reduce_unsupported=False,
        reduce_while_nodes=True,
        error_output_dir='./lt_graph_error')
    print(json_object)
    with open(lt_graph_json_file, 'w') as fid:
        json.dump(json_object, fid, sort_keys=True, indent=2)
        print(f'write json to {lt_graph_json_file}')
    cmdstr = 'python lt_sdk/visuals/plot_lgf_graph.py --pb_graph_path ' + lt_graph_path + ' --port [port]'
    print('to visualize in GUI, you can use:\n {}'.format(cmdstr))

def remove_dir_recursive(path):
        for item in os.listdir(path):
            path_file = os.path.join(path, item)
            print(path_file)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                remove_dir_recursive(path_file)
        os.removedirs(path)


class TFDaggerAdapter(ABC):
    def __init__(self, image_shape, input_name, output_name, train_scope):
        '''
        input_name must be same with ckpt_path
        '''
        self.image_shape = image_shape
        self.input_name = input_name
        self.output_name = output_name
        self.train_scope = train_scope
        self.net = None
        self.isess = None
        self.input_placeholder = None
        self.embeddings_placeholder = None
        self.restored = False
        self.adapted_to_dagger = False
    
    @abstractmethod
    def address_input_output_placeholder(self, *args, **kwargs):
        raise NotImplementedError("not implemented")

    @abstractmethod
    def preprocess_single_sample(self, *args, **kwargs):
        raise NotImplementedError('not implemented')

    def load_ckpt(self, ckpt_file, print_info=False):
        if self.restored:
            raise RuntimeError('model has been restored before')
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = tf.InteractiveSession(config=config)

        self.address_input_output_placeholder()

        # restore
        saver = tf.train.Saver()
        saver.restore(self.isess, ckpt_file)

        if print_info:
            tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            print(tensor_name_list)

        self.restored = True

    def load_pb_graphdef(self, pb_filepath, print_info=False):
        # if self.restored:
        #     raise RuntimeError('model has been restored before')
        with tf.gfile.FastGFile(pb_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        embed = tf.import_graph_def(graph_def, return_elements=[self.output_name+':0'])

        if print_info:
            tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            print(tensor_name_list)

        self.restored = True

    @classmethod
    def preprocess_func(batch_data):
        return batch_data

    def inference(self, image_data_or_path, print_info=False):
        if not self.restored:
            raise RuntimeError('model must been restored')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        image_data = self.preprocess_single_sample(image_data_or_path)
        try:
            # if import pb graphdef then if include import
            input_tensor = tf.get_default_graph().get_tensor_by_name('import/' + self.input_name + ':0')
            output_tensor = tf.get_default_graph().get_tensor_by_name('import/' + self.output_name + ':0')
        except Exception as e:
            # if import ckpt then include no import
            # TODO, actually, inference from restored ckpt is not OK by now
            input_tensor = tf.get_default_graph().get_tensor_by_name(self.input_name + ':0')
            output_tensor = tf.get_default_graph().get_tensor_by_name(self.output_name + ':0')    

        image_input = sess.run(image_data) # tensor to list
        embed_res = sess.run(output_tensor, feed_dict={input_tensor: image_input})  

        if print_info:
            print(f'image: {image_data_or_path}')
            print(f'embeddings:\n{embed_res}')
        return embed_res

    @abstractmethod
    def _adapt_to_dagger(self):
        raise NotImplementedError('_adapt_to_dagger no implemented')

    def freeze_graph_def(self, sess, input_graph_def, output_node_names):
        output_graph_def = None
        if self.train_scope in ['TF']:
            for node in input_graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in xrange(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            # Replace all the variables in the graph with constants of the same values
            output_graph_def = graph_util.convert_variables_to_constants(
                sess, input_graph_def, output_node_names)
        else:
            raise NotImplementedError('not implemented')
        return output_graph_def

    def save_to_saved_model(self, output_dir, need_adapt_to_dagger=False, force_rewrite=False):
        if not self.restored:
            raise RuntimeError('need resotore ckpt')
        if need_adapt_to_dagger and (not self.adapted_to_dagger):
            self._adapt_to_dagger()
            self.adapted_to_dagger = True
        if force_rewrite:
            if os.path.exists(output_dir):
                remove_dir_recursive(output_dir)
                print(f'remove dir {output_dir}')
        tf.saved_model.simple_save(self.isess, output_dir, inputs={self.input_name: self.input_placeholder}, 
            outputs={self.output_name: self.embeddings_placeholder})
        print(f'finished saved model to {output_dir}')

    def save_to_pb(self, output_path, need_adapt_to_dagger=False):
        if not self.restored:
            raise RuntimeError('need resotore ckpt')
        if need_adapt_to_dagger and (not self.adapted_to_dagger):
            self._adapt_to_dagger()
            self.adapted_to_dagger = True
        input_graph_def = self.isess.graph.as_graph_def()
        output_graph_def = self.freeze_graph_def(self.isess, input_graph_def, [self.output_name])
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with tf.gfile.GFile(output_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_path))

    def convert_to_lt_graph(self, input_model_path, output_lt_graph_path, input_type='TFGraphDef', calib_data=None, calib_sample_num=500):
        from lt_sdk.graph.transform_graph import utils as lt_utils       
        def calib_infer_process(light_graph, calibration_data2, config):
            outputs = lt.run_functional_simulation(light_graph, calibration_data2, config)
            for inf_out in outputs.batches:
                for named_ten in inf_out.results:
                    if named_ten.edge_info.name.startswith(self.output_name):
                        embed_res = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                        try:
                            embed_res = embed_res.numpy()
                        except Exception:
                            with tf.Session() as sess:
                                embed_res = sess.run(embed_res) # tensor to list
            return embed_res, light_graph

        import lt_sdk as lt
        from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
        config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=eval('graph_types_pb2.'+input_type))
        input_graph = lt.import_graph(input_model_path, config)
        if calib_data is None:
            print('no data for calibration when convert to lt graph')
            lt_graph = lt.transform_graph(input_graph, config)
            lt.export_graph(lt_graph, output_lt_graph_path, config)
        # elif calib_data == 'mnist':
        #     print('use {} samples of mnist for calibration when convert to lt graph'.format(calib_sample_num))
        #     from tensorflow.examples.tutorials.mnist import input_data
        #     mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        #     batch_size_test = 1 # stable as 1
        #     total_batch_test = int(mnist.test.num_examples / batch_size_test )
        #     for ibatch in range(total_batch_test):
        #         print('ibatch = {}'.format(ibatch))
        #         batch_x, _ = mnist.test.next_batch(batch_size_test)
        #         named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [batch_x])
        #         batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size_test)
        #         lt_graph = lt.transform_graph(input_graph, config, calibration_data=batch_input)
        #         res, lt_graph2 = calib_infer_process(lt_graph, batch_input, config)
        #         if (ibatch+1) * batch_size_test >= calib_sample_num:
        #             lt.export_graph(lt_graph2, output_lt_graph_path, config)
        #             break
        elif calib_data == 'mnist':
            print('use {} samples of mnist for calibration when convert to lt graph'.format(calib_sample_num))
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            total_batch_test = int(mnist.test.num_examples / calib_sample_num )
            for ibatch in range(total_batch_test):
                print('ibatch = {}'.format(ibatch))
                batch_x, _ = mnist.test.next_batch(calib_sample_num)
                named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [batch_x])
                batch_input = lt.data.batch.batch_inputs(named_tensor, calib_sample_num)
                lt_graph = lt.transform_graph(input_graph, config, calibration_data=batch_input)
                lt.export_graph(lt_graph, output_lt_graph_path, config, graph_types_pb2.LGFProtobuf)
                break     
        # elif calib_data == 'cifar10':
        #     print('use {} samples of mnist for calibration when convert to lt graph')
        #     cifar10 = tf.keras.datasets.cifar10
        #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #     x_train = x_train[:, :, :, :] / 255.
        #     x_test = x_test[:, :, :, :] / 255.
        #     y_train = np.squeeze(y_train, axis=1)
        #     y_test = np.squeeze(y_test, axis=1)
        #     batch_size_test = 1
        #     test_data_loader = NpDataloader(x_test, y_test, batch_size_test, None, True)
        #     ibatch = 0
        #     for batch_x, _ in test_data_loader:
        #         print('ibatch = {}'.format(ibatch))
        #         named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [batch_x])
        #         batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size_test)
        #         lt_graph = lt.transform_graph(input_graph, config, calibration_data=batch_input)
        #         _, lt_graph2 = calib_infer_process(lt_graph, batch_input, config)
        #         if (ibatch+1) * batch_size_test >= calib_sample_num:
        #             lt.export_graph(lt_graph2, output_lt_graph_path, config)
        #             break
        #         ibatch += 1
        elif calib_data == 'cifar10':
            print('use {} samples of cifar for calibration when convert to lt graph'.format(calib_sample_num))
            cifar10 = tf.keras.datasets.cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train = x_train[:, :, :, :] / 255.
            x_test = x_test[:, :, :, :] / 255.
            y_train = np.squeeze(y_train, axis=1)
            y_test = np.squeeze(y_test, axis=1)
            test_data_loader = NpDataloader(x_test, y_test, calib_sample_num, None, True)
            ibatch = 0
            for batch_x, _ in test_data_loader:
                print('ibatch = {}'.format(ibatch))
                named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [batch_x])
                batch_input = lt.data.batch.batch_inputs(named_tensor, calib_sample_num)
                lt_graph = lt.transform_graph(input_graph, config, calibration_data=batch_input)
                debug_calib = False
                if debug_calib:
                    lt_graph_nocalibdata = lt.transform_graph(input_graph, config)
                    image_data = self.preprocess_single_sample('data/datasets/cifar10/imgs/test/010044.png')
                    try:
                        image_data = image_data.numpy()
                    except Exception:
                        with tf.Session() as sess:
                            image_data = sess.run(image_data) # tensor to list
                    named_tensor2 = lt.data.named_tensor.NamedTensorSet([self.input_name], [image_data])
                    batch_input2 = lt.data.batch.batch_inputs(named_tensor2, 1)
                    ori_res2 = calib_infer_process(input_graph, batch_input2, config)
                    trans_calib_res2, _ = calib_infer_process(lt_graph, batch_input2, config)
                    trans_nocalibdata_res2, _ = calib_infer_process(lt_graph_nocalibdata, batch_input2, config)
                    print('ori_res2: {}'.format(ori_res2[0]))
                    print('trans_calib_res2: {}'.format(trans_calib_res2[0]))
                    print('trans_nocalibdata_res2: {}'.format(trans_nocalibdata_res2[0]))
                
                lt.export_graph(lt_graph, output_lt_graph_path, config, graph_types_pb2.LGFProtobuf)
                break
        
        else:
            raise RuntimeError('unsupported calib_data')
        return

    def lt_func_infererence(self, lt_graph_path, image_data_or_path, input_type='TFGraphDef', print_info=False):
        import lt_sdk as lt
        from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
        from lt_sdk.graph.transform_graph import utils as lt_utils

        def infer_process(light_graph=None, inputs=None, config=None):
            outputs = lt.run_functional_simulation(light_graph, inputs, config)
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
        graph = lt.import_graph(lt_graph_path, config, graph_types_pb2.LGFProtobuf)

        image_data = self.preprocess_single_sample(image_data_or_path)
        try:
            image_data = image_data.numpy()
        except Exception:
            with tf.Session() as sess:
                image_data = sess.run(image_data) # tensor to list

        batch_size = 1
        named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [image_data])
        batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size)
        embed_res = infer_process(graph, batch_input, config)
        if print_info:
            print(f'embeddings:\n{embed_res}')
        return embed_res


    def lt_perf_infererence(self, lt_graph_path, output_trace_path, input_type='TFGraphDef', print_info=False):
        import lt_sdk as lt
        from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
        from lt_sdk.graph.transform_graph import utils as lt_utils

        config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=eval('graph_types_pb2.'+input_type))
        graph = lt.import_graph(lt_graph_path, config, graph_types_pb2.LGFProtobuf)
        execstat = lt.run_performance_simulation(graph, config)
        from lt_sdk.visuals import sim_result_to_trace
        # print(execstat)
        sim_result_to_trace.instruction_trace(output_trace_path, execstat, config.hw_specs, config.sim_params)
