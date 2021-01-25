import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np

from models import ResNet128
from utils import init_conv_weight, init_attention_weight, init_res_weight, smart_res_block
from utils import init_fc_weight, smart_conv_block, smart_fc_block, smart_atten_block, swish

FLAGS = flags.FLAGS


class DualModelWithTop(object):
    """
    The model that combines the two raw image processing Convolutional models and feeds their outputs to
    the Discriminator or a dense layer (FC block).
    """

    def __init__(self, model_a, model_b, top, pretrained=True):
        self.pretrained = pretrained
        self.model_a = model_a
        self.model_b = model_b
        self.top = top

    def _get_submodel_scopes(self, scope):
        if len(scope) == 0:
            sub_scope = 'dual'
        else:
            sub_scope = scope + '/dual'
        if self.pretrained:
            return scope, scope, sub_scope + '_top'
        else:
            return sub_scope + '_conv', sub_scope + '_conv', sub_scope + '_top'

    def construct_weights(self, scope=''):
        scope_a, scope_b, scope_top = self._get_submodel_scopes(scope)
        weights = {}
        if not self.pretrained:
            weights.update(self.model_a.construct_weights(scope=scope_a))
        # weights.update(self.model_b.construct_weights(scope=scope_b)) # a and b use identical weights
        weights.update(self.top.construct_weights(scope=scope_top))
        return weights

    def forward(self, inp_a, inp_b, weights, attention_mask=None, reuse=False, scope='', stop_grad=False, label=None,
                stop_at_grad=False, stop_batch=False, latent=None):
        scope_a, scope_b, scope_top = self._get_submodel_scopes(scope)
        a = self.model_a.forward(inp_a, weights, attention_mask, reuse, scope_a, stop_grad, label, stop_at_grad,
                                 stop_batch, latent)
        b = self.model_b.forward(inp_b, weights, attention_mask, reuse, scope_b, stop_grad, label, stop_at_grad,
                                 stop_batch, latent)
        z = tf.keras.layers.concatenate([a, b])
        energy = self.top.forward(z, weights, attention_mask, reuse, scope_top, stop_grad, label, stop_at_grad, stop_batch,
                                  latent)
        return energy


class Discriminator(object):
    """
    Just a dense layer to be used as the last model that outputs energy.
    Input is the concatenation of 2 Convolutional layer outputs.
    """

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


class ResNet128NoTop(ResNet128):
    """
    Convolutional layer model suitable for using pretrained model
    """

    def __init__(self, num_channels=3, num_filters=64, train=False, classes=1000):
        super().__init__(num_channels, num_filters, train, classes)

    # use construct_weights of superclass, no override

    def forward(self, inp, weights, attention_mask=None, reuse=False, scope='', stop_grad=False, label=None, stop_at_grad=False, stop_batch=False, latent=None):
        weights = weights.copy()
        batch = tf.shape(inp)[0]

        if not FLAGS.cclass:
            label = None


        if stop_grad:
            for k, v in weights.items():
                if type(v) == dict:
                    v = v.copy()
                    weights[k] = v
                    for k_sub, v_sub in v.items():
                        v[k_sub] = tf.stop_gradient(v_sub)
                else:
                    weights[k] = tf.stop_gradient(v)

        if FLAGS.swish_act:
            act = swish
        else:
            act = tf.nn.leaky_relu

        dropout = self.dropout
        train = self.train

        # Make sure gradients are modified a bit
        inp = smart_conv_block(inp, weights, reuse, 'c1_pre', use_stride=False, activation=act)
        hidden1 = smart_res_block(inp, weights, reuse, 'res_optim', label=label, dropout=dropout, train=train, downsample=True, adaptive=False)

        if FLAGS.use_attention:
            hidden1 = smart_atten_block(hidden1, weights, reuse, 'atten', stop_at_grad=stop_at_grad)

        hidden2 = smart_res_block(hidden1, weights, reuse, 'res_3', stop_batch=stop_batch, downsample=True, adaptive=True, label=label, dropout=dropout, train=train, act=act)
        hidden3 = smart_res_block(hidden2, weights, reuse, 'res_5', stop_batch=stop_batch, downsample=True, adaptive=True, label=label, dropout=dropout, train=train, act=act)
        hidden4 = smart_res_block(hidden3, weights, reuse, 'res_7', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=True, adaptive=True)
        hidden5 = smart_res_block(hidden4, weights, reuse, 'res_9', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=True, adaptive=False)
        hidden6 = smart_res_block(hidden5, weights, reuse, 'res_10', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=False, adaptive=False)

        if FLAGS.swish_act:
            hidden6 = act(hidden6)
        else:
            hidden6 = tf.nn.relu(hidden6)

        return hidden6
