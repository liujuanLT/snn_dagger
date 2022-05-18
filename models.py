import tensorflow as tf
import tensorflow.contrib.slim as slim

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

def fc_lif_net(inputs, lif_state1, lif_state2, is_training, tau, reuse=None, scope='fcLif'):    
    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=None,
                        biases_initializer = None,
                        normalizer_fn=None):
        with tf.variable_scope(scope, 'fcLif', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
                    # net0 = slim.flatten(inputs, scope='flatten0')
                    net1 = slim.fully_connected(inputs, 512, activation_fn=None, scope='fc0')
                    net, new_lif_state1 = LIFNode(net1, lif_state=lif_state1, tau=tau)
                    net2 = slim.fully_connected(net, 10, activation_fn=None, scope='fc1')
                    p2, new_lif_state2 = LIFNode(net2, lif_state=lif_state2, tau=tau)
                 
            return p2, new_lif_state1, new_lif_state2

def fc_lif_net_clock_A(inputs, is_training, T=10, tau=2.0, reuse=None, scope='fcLif'):
    for t in range(T):
        lessthan = (tf.random_uniform(tf.shape(inputs)) < inputs)
        encoded_inputs = tf.cast(lessthan, tf.float32, name='encodes_inputs')
        if t == 0:
            num_spikes_tensor, new_lif_state_tensor1, new_lif_state_tensor2 = fc_lif_net(encoded_inputs, lif_state1=None, lif_state2=None, tau=tau, is_training=is_training, reuse=reuse, scope=scope)
            out_spikes_counter_tensor = num_spikes_tensor
        else:
            num_spikes_tensor, new_lif_state_tensor1, new_lif_state_tensor2 = fc_lif_net(encoded_inputs, lif_state1=new_lif_state_tensor1, lif_state2=new_lif_state_tensor2, tau=tau, is_training=is_training, reuse=reuse, scope=scope)
            out_spikes_counter_tensor += num_spikes_tensor
    return out_spikes_counter_tensor

# def fc_lif_net_clock_A(inputs, is_training, T=10, tau=2.0, reuse=None, scope='fcLif'):
#     shape_x = tf.shape(inputs, name='shape_x')
#     for t in range(T):
#         lessthan = tf.less(tf.random_uniform(shape_x), inputs, name='lessthan')
#         encoded_inputs = tf.cast(lessthan, tf.float32, name='encodes_inputs')
#         if t == 0:
#             num_spikes_tensor, new_lif_state_tensor1, new_lif_state_tensor2 = fc_lif_net(encoded_inputs, lif_state1=None, lif_state2=None, tau=tau, is_training=is_training, reuse=reuse, scope=scope)
#             out_spikes_counter_tensor = num_spikes_tensor
#         else:
#             num_spikes_tensor, new_lif_state_tensor1, new_lif_state_tensor2 = fc_lif_net(encoded_inputs, lif_state1=new_lif_state_tensor1, lif_state2=new_lif_state_tensor2, tau=tau, is_training=is_training, reuse=reuse, scope=scope)
#             out_spikes_counter_tensor += num_spikes_tensor
#     return out_spikes_counter_tensor


def fc_lif_net_clock_B(inputs, is_training, T=10, tau=2.0, reuse=None, scope='fcLif'):
    all_encoded_inputs = []
    shape_x = tf.shape(inputs, name='shape_x')
    for t in range(T):
        lessthan = tf.less(tf.random_uniform(shape_x),  inputs, name='lessthan')
        encoded_inputs = tf.cast(lessthan, tf.float32, name='encodes_inputs')
        all_encoded_inputs.append(encoded_inputs)
    for t in range(T):
        lessthan = (tf.random_uniform(tf.shape(inputs)) < inputs)
        encoded_inputs = tf.cast(lessthan, tf.float32, name='encodes_inputs')
        if t == 0:
            num_spikes_tensor, new_lif_state_tensor1, new_lif_state_tensor2 = fc_lif_net(all_encoded_inputs[t], lif_state1=None, lif_state2=None, tau=tau, is_training=is_training, reuse=reuse, scope=scope)
            out_spikes_counter_tensor = num_spikes_tensor
        else:
            num_spikes_tensor, new_lif_state_tensor1, new_lif_state_tensor2 = fc_lif_net(all_encoded_inputs[t], lif_state1=new_lif_state_tensor1, lif_state2=new_lif_state_tensor2, tau=tau, is_training=is_training, reuse=reuse, scope=scope)
            out_spikes_counter_tensor += num_spikes_tensor
    return out_spikes_counter_tensor
