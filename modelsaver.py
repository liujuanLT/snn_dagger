import tensorflow as tf
from abc import ABC, abstractclassmethod, abstractmethod
from six.moves import xrange  # @UnresolvedImport
from tensorflow.python.framework import graph_util
import numpy as np
import os
from models import fc_lif_net_clock_A, fc_lif_net_clock_B

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

    def load_ckpt(self, ckpt_file, print_info=False):
        if self.restored:
            raise RuntimeError('model has been restored before')
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = tf.InteractiveSession(config=config)

        if self.train_scope == 'snn_clock_mnist':
            self.input_placeholder = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_name)
            out_spikes_counter_tensor = self.net(self.input_placeholder, is_training=False, T=10, tau=2., reuse=tf.AUTO_REUSE)
            self.embeddings_placeholder = tf.add(0.0, out_spikes_counter_tensor, name=self.output_name)
        else:
            raise NotImplementedError('not implemented')

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
        if self.train_scope in ['snn_clock_mnist']:
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
        else:
            raise NotImplementedError('not implemented')

        if print_info:
            print(f'image: {image_data_or_path}')
            print(f'embeddings:\n{embed_res}')
        return embed_res

    @abstractmethod
    def _adapt_to_dagger(self):
        raise NotImplementedError('_adapt_to_dagger no implemented')

    def freeze_graph_def(self, sess, input_graph_def, output_node_names):
        output_graph_def = None
        if self.train_scope in ['snn_clock_mnist']:
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

    @classmethod
    def convert_to_lt_graph(cls, input_model_path, output_lt_graph_path, input_type='TFGraphDef'):
        import lt_sdk as lt
        from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
        config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=eval('graph_types_pb2.'+input_type))
        input_graph = lt.import_graph(input_model_path, config)
        trans_graph = lt.transform_graph(input_graph, config)
        lt.export_graph(trans_graph, output_lt_graph_path, config)
        return

    def lt_func_infererence(self, lt_graph_path, image_path, input_type='TFGraphDef', print_info=False):
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

        if self.train_scope in ['snn_clock_mnist']:
            try:
                raw_image = tf.io.read_file(image_path, 'r') # for tf 1.15
            except Exception as e:
                raw_image = tf.read_file(image_path, 'r') # for tf 1.7, there is no tf.io
            image_data = tf.image.decode_png(raw_image, channels=1)
            # image_data = tf.expand_dims(image_data, 0)
            image_data = tf.reshape(image_data, [1, 784])
            try:
                image_data = image_data.numpy()
            except Exception:
                with tf.Session() as sess:
                    image_data = sess.run(image_data) # tensor to list
        else:
            raise NotImplementedError('not implemented')

        batch_size = 1
        named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [image_data])
        batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size)
        # embed_res = lt.run_functional_simulation(graph, batch_input, config)
        embed_res = infer_process(graph, batch_input, config)
        if print_info:
            print(f'embeddings:\n{embed_res}')
        return embed_res


    def lt_perf_infererence(self, lt_graph_path, output_trace_path, input_type='TFGraphDef', print_info=False):
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
        execstat = lt.run_performance_simulation(graph, config)
        from lt_sdk.visuals import sim_result_to_trace
        # print(execstat)
        sim_result_to_trace.instruction_trace(output_trace_path, execstat, config.hw_specs, config.sim_params)

                
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
        graph = lt.import_graph(lt_graph_path, config)

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

class SnnClockMnistModelSaverA(TFDaggerAdapter):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'snn_clock_mnist')
        self.net = fc_lif_net_clock_A

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')

class SnnClockMnistModelSaverB(TFDaggerAdapter):
    def __init__(self, image_shape):
        super().__init__(image_shape, 'image_batch', 'logits', 'snn_clock_mnist')
        self.net = fc_lif_net_clock_B

    def _adapt_to_dagger(self):
        if not self.restored:
            raise RuntimeError('model must been restored')
            
def test_snn_clock_mnist_A():
    modelsaver = SnnClockMnistModelSaverA((28,28))
    model_dir_path = 'data/snn_trained_model/snn_clock_mnist_2022-05-18-00-34'
    ckpt_file = os.path.join(model_dir_path, 'snn_clock_mnist.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_clock_mnist.pb')
    out_pb_file_adaptlt = os.path.join(model_dir_path, 'snn_clock_mnist_adaptlt.pb')
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
    ltgraph_file = os.path.join(model_dir_path, 'snn_clock_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel')
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(model_dir_path, 'snn_clock_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(model_dir_path, 'snn_clock_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')

            
def test_snn_clock_mnist_B():
    modelsaver = SnnClockMnistModelSaverB((28,28))
    model_dir_path = 'data/snn_trained_model/snn_clock_mnist_2022-05-18-12-15'
    ckpt_file = os.path.join(model_dir_path, 'snn_clock_mnist.ckpt')
    saved_model_dir = os.path.join(model_dir_path, 'saved_model')
    out_pb_file = os.path.join(model_dir_path, 'snn_clock_mnist.pb')
    out_pb_file_adaptlt = os.path.join(model_dir_path, 'snn_clock_mnist_adaptlt.pb')
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
    ltgraph_file = os.path.join(model_dir_path, 'snn_clock_mnist_ltgraph.pb')
    modelsaver.convert_to_lt_graph(saved_model_dir, ltgraph_file, input_type='TFSavedModel')
    embed_res_lt = modelsaver.lt_func_infererence(ltgraph_file, image_path, input_type='TFSavedModel', print_info=True)
    cls_lt = np.argmax(embed_res_lt, axis=1)[0]
    print(f'class={cls_lt}')
    ltgraph_json_file = os.path.join(model_dir_path, 'snn_clock_mnist_ltgraph.json')
    visual_lt(ltgraph_file, ltgraph_json_file)
    output_trace_path = os.path.join(model_dir_path, 'snn_clock_mnist_ltgraph.trace')
    modelsaver.lt_perf_infererence(ltgraph_file, output_trace_path)
    # note, it's quite time-consuming, takes about 5 hours
    acc_lt = modelsaver.lt_mnist_acc_test(ltgraph_file, input_type='TFSavedModel', resfile=os.path.join(model_dir_path, 'mnist_acc_test_ltgraph.txt'))
    print(f'acc={acc_lt}')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    test_snn_clock_mnist_A()
    test_snn_clock_mnist_B()