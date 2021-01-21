import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from utils import init_conv_weight, init_attention_weight, init_res_weight, smart_res_block
from utils import init_fc_weight, smart_conv_block, smart_fc_block, smart_atten_block, swish

FLAGS = flags.FLAGS


class DualModelWithTop(object):
    def __init__(self, model_a, model_b, top):
        self.model_a = model_a
        self.model_b = model_b
        self.top = top

    def construct_weights(self, scope=''):
        weights = {}
        if len(scope) == 0:
            scope = 'dual'
        else:
            scope += '_dual'
        weights.update(self.model_a.construct_weights(scope=scope))
        weights.update(self.model_b.construct_weights(scope=scope))
        weights.update(self.top.construct_weights(scope=scope))
        return weights

    def forward(self, inp_a, inp_b, weights, attention_mask=None, reuse=False, scope='', stop_grad=False, label=None,
                stop_at_grad=False, stop_batch=False, latent=None):
        if len(scope) == 0:
            scope = 'dual'
        else:
            scope += '_dual'
        a = self.model_a.forward(inp_a, weights, attention_mask, reuse, scope, stop_grad, label, stop_at_grad,
                                 stop_batch, latent)
        b = self.model_b.forward(inp_b, weights, attention_mask, reuse, scope, stop_grad, label, stop_at_grad,
                                 stop_batch, latent)
        z = tf.keras.layers.concatenate([a, b])
        energy = self.top.forward(z, weights, attention_mask, reuse, scope, stop_grad, label, stop_at_grad, stop_batch, latent)
        return energy


class Discriminator(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, num_filters=64):
        self.dim_hidden = num_filters

    def construct_weights(self, scope=''):
        weights = {}

        with tf.variable_scope(scope):
            init_fc_weight(weights, 'fc5', 2 * 8 * self.dim_hidden, 1, spec_norm=False)  # *2 because two input models
            init_attention_weight(weights, 'atten', self.dim_hidden, self.dim_hidden / 2., trainable_gamma=True)

        return weights

    def forward(self, inp, weights, attention_mask=None, reuse=False, scope='', stop_grad=False, label=None,
                stop_at_grad=False, stop_batch=False, latent=None):
        weights = weights.copy()

        if stop_grad:
            for k, v in weights.items():
                if type(v) == dict:
                    v = v.copy()
                    weights[k] = v
                    for k_sub, v_sub in v.items():
                        v[k_sub] = tf.stop_gradient(v_sub)
                else:
                    weights[k] = tf.stop_gradient(v)

        hidden5 = tf.reduce_sum(inp, [1, 2])
        hidden6 = smart_fc_block(hidden5, weights, reuse, 'fc5')
        energy = hidden6
        return energy
