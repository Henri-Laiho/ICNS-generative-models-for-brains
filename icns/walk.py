import os
import random

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from icns.dual_model import DualModelWithTop, ResNet128NoTop, Discriminator
from icns.identity_data import CelebAPairs

import numpy as np
from tensorflow.python.platform import flags

from models import ResNet128
from baselines.logger import TensorBoardOutputFormat
from utils import average_gradients, optimistic_restore
from torch.utils.data import DataLoader
import torch
from custom_adam import AdamOptimizer

torch.manual_seed(0)
np.random.seed(0)
tf.set_random_seed(0)

FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('datasource', 'random',
                    'initialization for chains, either random or default (decorruption)')
flags.DEFINE_string('dataset', 'cubes',
                    'concept combination (cubes, pairs, pos, continual, color, or cross right now)')
flags.DEFINE_integer('batch_size', 16, 'Size of inputs')
flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')
flags.DEFINE_integer('data_workers', 4,
                     'Number of different data workers to load data in parallel')
flags.DEFINE_integer('cond_idx', 0, 'By default, train conditional models on conditioning on position')

# General Experiment Settings
flags.DEFINE_string('logdir', 'cachedir',
                    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 250, 'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000, 'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 10000, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 2.1e-4, 'Learning for training')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')

# EBM Specific Experiments Settings
flags.DEFINE_float('ml_coeff', 1.0, 'Maximum Likelihood Coefficients')
flags.DEFINE_float('l2_coeff', 1.0, 'L2 Penalty training')
flags.DEFINE_bool('cclass', True, 'Whether to conditional training in models')
flags.DEFINE_bool('model_cclass', False, 'use unsupervised clustering to infer fake labels')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')
flags.DEFINE_string('objective', 'cd', 'use either contrastive divergence objective(least stable),'
                                       'logsumexp(more stable)'
                                       'softplus(most stable)')
flags.DEFINE_bool('zero_kl', True, 'whether to zero out the kl loss')
flags.DEFINE_float('keep_ratio', 0.05, 'Ratio of things to keep')
flags.DEFINE_bool('fft', False, 'Run all steps of model on the Fourier domain instead of image domain')
flags.DEFINE_bool('augment_vis', False, 'Augmentations on images to improve smoothness')

# Setting for MCMC sampling
flags.DEFINE_float('proj_norm', 0.0, 'Maximum change of input images')
flags.DEFINE_string('proj_norm_type', 'li', 'Either li or l2 ball projection')
flags.DEFINE_integer('num_steps', 40, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 100, 'Size of steps for gradient descent')
flags.DEFINE_float('attention_lr', 1e5, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', True, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_float('noise_scale', 1., 'Relative amount of noise for MCMC')
flags.DEFINE_bool('pcd', False, 'whether to use pcd training instead')

# Architecture Settings
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('large_model', False, 'whether to use a large model')
flags.DEFINE_bool('larger_model', False, 'Deeper ResNet32 Network')
flags.DEFINE_bool('wider_model', False, 'Wider ResNet32 Network')

# Dataset settings
flags.DEFINE_bool('mixup', False, 'whether to add mixup to training images')
flags.DEFINE_bool('augment', False, 'whether to augmentations to images')
flags.DEFINE_float('rescale', 1.0, 'Factor to rescale inputs from 0-1 box')
flags.DEFINE_integer('celeba_cond_idx', 1, 'conditioned index to select the celeba model')

# Concept combination experiments
flags.DEFINE_bool('comb_mask', False, 'condition of combinations')
flags.DEFINE_integer('cond_func', 3, 'Number of seperate conditional masks to use')
flags.DEFINE_bool('heir_mask', False, 'training a conditional model on distance on attention mask')

# Settings for antialiasing images?
flags.DEFINE_bool('antialias', False, 'whether to antialias the image before feeding it in')

# Flags for joint learning of model with other model
flags.DEFINE_bool('prelearn_model', True, 'whether to load a prelearned model')
flags.DEFINE_string('prelearn_exp', 'celeba_attractive', 'prelearned model name')
flags.DEFINE_integer('prelearn_iter', 22000, 'iteration of the experiment')
flags.DEFINE_integer('prelearn_label', 2, 'number of labels for the training')

# Cross product experiments settings
flags.DEFINE_bool('cond_size', False, 'condition of color ')
flags.DEFINE_bool('cond_pos', False, 'condition of position loc')
flags.DEFINE_float('ratio', 1.0, 'ratio of data to keep')
flags.DEFINE_bool('joint_baseline', False, 'use a joint baseline to train models')

flags.DEFINE_string('savedir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('task', 'negation_figure', 'conceptcombine, combination_figure, negation_figure, or_figure')

flags.DEFINE_bool('latent_energy', False, 'latent energy in model')
flags.DEFINE_bool('proj_latent', False, 'Projection of latents')

# ICNS
flags.DEFINE_integer('samples_per_ground', 48, 'for how many samples we should keep the other image constant')
flags.DEFINE_integer('cycles_per_side', 2,
                     'number of samples_per_ground cycles before swapping the side of the constant image')
flags.DEFINE_integer('min_occurrences', 5,
                     'if there are less than this many occurrences of a celebrity in the dataset, then drop all images of that celebrity')
flags.DEFINE_float('pos_neg_balance', 0.5,
                   'the balance of negative and positive image pairs: 1.0=100% positive, 0.5 = 50/50, 0.0=100% negative')

FLAGS.step_lr = FLAGS.step_lr * FLAGS.rescale
FLAGS.swish_act = True

FLAGS.batch_size *= FLAGS.num_gpus

print("{} batch size, __name__={}".format(FLAGS.batch_size, __name__))


def walk_single(z1, z2, target_vars, sess, step_length=0.1, threshold_crossing_limit=0, return_energy=False,
                threshold_energy=0.1):
    LABEL_POS = target_vars['LABEL_POS']
    energy_z = target_vars['energy_z']
    output = [energy_z]
    Z = target_vars['Z']
    label = [[0, 1]]
    dz = z2 - z1
    dz_magnitude = np.linalg.norm(dz)
    dzw = dz / dz_magnitude * step_length
    n_steps = int(dz_magnitude / step_length)
    print('n_steps=%d' % n_steps, 'dz_magnitude=%f' % dz_magnitude)

    zw = z1
    over_threshold = 0
    energy_sum = 0

    for i in range(n_steps):
        z = np.concatenate((z1, zw))
        energy = sess.run(output, {Z: [z], LABEL_POS: label})[0][0]
        energy_sum += energy
        if not return_energy and energy > threshold_energy:
            over_threshold += energy - threshold_energy
            if over_threshold > threshold_crossing_limit:
                return False
        zw = zw + dzw

    z = np.concatenate((z1, z2))
    energy = sess.run(output, {Z: [z], LABEL_POS: label})[0][0]
    energy_sum += energy
    if not return_energy and energy > threshold_energy:
        over_threshold += energy - threshold_energy
        if over_threshold > threshold_crossing_limit:
            return False
    return energy_sum / n_steps if return_energy else True


def test(target_vars, saver, sess, logger, dataset):
    X1 = target_vars['X1']
    X2 = target_vars['X2']
    Z1 = target_vars['Z1']
    Z2 = target_vars['Z2']
    LABEL = target_vars['LABEL']
    LABEL_POS = target_vars['LABEL_POS']
    x_mod1 = target_vars['test_x_mod1']
    x_mod2 = target_vars['test_x_mod2']
    energy_pos = target_vars['energy_pos']

    output = [energy_pos, Z1, Z2]

    direct_preds = []
    walk_preds = []
    y_true = []
    direct_energies = []
    walk_energies = []

    test_size = 10000
    rand = random.Random(x=10)
    testset_index = rand.choices(range(len(dataset)), k=test_size)

    for i, idx in enumerate(testset_index):
        print('\rtesting: %d/%d ' % (i + 1, test_size), end='')
        # (img_corrupt1, img_corrupt2), (img1, img2), label = dataset[i]
        _, (img1, img2), label = dataset[idx]

        direct_energy, z1, z2 = sess.run(output, {X1: [img1], X2: [img2], LABEL_POS: [label]})
        walk_energy = walk_single(z1[0], z2[0], target_vars, sess, step_length=0.001, return_energy=True)

        direct_preds.append(direct_energy[0][0] <= 0.15)
        direct_energies.append(direct_energy[0][0])
        walk_preds.append(walk_energy <= 0.15)
        walk_energies.append(walk_energy)
        y_true.append(label[1])

    direct_preds = np.array(direct_preds)
    walk_preds = np.array(walk_preds)
    y_true = np.array(y_true)
    direct_energies = np.array(direct_energies)
    walk_energies = np.array(walk_energies)

    print('actual positives:', sum(y_true))
    print('direct prediction accuracy:', accuracy_score(y_true, direct_preds), 'f1-score:',
          f1_score(y_true, direct_preds))
    print('direct prediction positives:', sum(direct_preds))
    print('direct prediction positives mean energy:', np.mean(direct_energies[y_true == 1]), 'negatives:',
          np.mean(direct_energies[y_true == 0]))
    print('walk accuracy:', accuracy_score(y_true, walk_preds), 'walk f1-score:', f1_score(y_true, walk_preds))
    print('walk positives:', sum(walk_preds))
    pos = walk_energies[y_true == 1]
    pos = pos[pos < 1E308]
    pos = pos[pos > -1E308]
    neg = walk_energies[y_true == 0]
    neg = neg[neg < 1E308]
    neg = neg[neg > -1E308]
    print('walk positives mean energy:', np.mean(pos), 'negatives:', np.mean(neg))


def main():
    logdir = os.path.join(FLAGS.logdir, FLAGS.exp)
    print('logging in', logdir)
    logger = TensorBoardOutputFormat(logdir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.74)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    label_size = None

    sess = tf.Session(config=config)
    LABEL = None

    models_orig = ['celeba_smiling', 'celeba_male', 'celeba_attractive', 'celeba_black', 'celeba_old',
                   'celeba_wavy_hair', 'celeba_old']
    resume_iters_orig = [24000, 23000, 22000, 32000, 24000, 24000, 24000]

    assert FLAGS.prelearn_model
    # models = [models_orig[2]]
    models = [FLAGS.prelearn_exp]
    # resume_iters = [resume_iters_orig[2]]
    resume_iters = [FLAGS.prelearn_iter]
    select_idx = [1]

    print("Loading data...")
    if FLAGS.dataset == 'celeba':
        dataset = CelebAPairs(samples_per_ground=FLAGS.samples_per_ground,
                              cycles_per_side=FLAGS.cycles_per_side,
                              minimum_occurrences=FLAGS.min_occurrences,
                              pos_probability=FLAGS.pos_neg_balance, random_state=0)
        test_dataset = dataset
        channel_num = 3
        X_NOISE1 = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
        X_NOISE2 = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
        X1 = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
        X2 = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
        Z = tf.placeholder(shape=(None, 64 * 8 * 2), dtype=tf.float32)
        LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)

        restore_model = ResNet128(
            num_channels=channel_num,
            num_filters=64,
            classes=2)
        model = DualModelWithTop(
            ResNet128NoTop(
                num_channels=channel_num,
                num_filters=64,
                classes=2),
            ResNet128NoTop(
                num_channels=channel_num,
                num_filters=64,
                classes=2), Discriminator())
    else:
        raise NotImplementedError

    if FLAGS.joint_baseline:
        raise NotImplementedError
    else:
        print("label size here ", label_size)
        channel_num = 3
        HEIR_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        ATTENTION_MASK = tf.placeholder(shape=(None, 64, 64, FLAGS.cond_func), dtype=tf.float32)

        if FLAGS.dataset != "celeba":
            raise NotImplementedError
        else:
            # Finish initializing all variables
            sess.run(tf.global_variables_initializer())

        if FLAGS.heir_mask:
            raise NotImplementedError

        print("Done loading...")

        # Load pretrained weights
        # weights = restore_model.construct_weights('context_0')

        # Now go load the correct files
        '''for i, (model_name, resume_iter) in enumerate(zip(models, resume_iters)):
            # Model 1 will be conditioned on size
            save_path_size = os.path.join(FLAGS.logdir, model_name, 'model_{}'.format(resume_iter))
            v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(i))
            v_map = {(v.name.replace('context_{}'.format(i), 'context_0')[:-2]): v for v in v_list}
            saver = tf.train.Saver(v_map)
            saver.restore(sess, save_path_size)'''

        # weights.update(model.construct_weights('context_0'))
        weights = model.construct_weights('context_0')

        if FLAGS.heir_mask:
            raise NotImplementedError

        X_SPLIT1 = tf.split(X1, FLAGS.num_gpus)
        X_SPLIT2 = tf.split(X2, FLAGS.num_gpus)
        X_NOISE_SPLIT1 = tf.split(X_NOISE1, FLAGS.num_gpus)
        X_NOISE_SPLIT2 = tf.split(X_NOISE2, FLAGS.num_gpus)
        LABEL_SPLIT = tf.split(LABEL, FLAGS.num_gpus)
        LABEL_POS_SPLIT = tf.split(LABEL_POS, FLAGS.num_gpus)
        attention_mask = ATTENTION_MASK
        tower_grads = []

        optimizer = AdamOptimizer(FLAGS.lr, beta1=0.0, beta2=0.99)

        target_vars = {}
        for j in range(FLAGS.num_gpus):
            x_mod1 = X_SPLIT1[j]
            x_mod2 = X_SPLIT2[j]

            Z1 = model.model_a.forward(
                X1,
                weights,
                attention_mask,
                label=LABEL_POS_SPLIT[j],
                stop_at_grad=False)
            Z2 = model.model_b.forward(
                X2,
                weights,
                attention_mask,
                label=LABEL_POS_SPLIT[j],
                stop_at_grad=False)
            energy_z = model.top.forward(
                Z,
                weights,
                attention_mask,
                label=LABEL_POS_SPLIT[j],
                stop_at_grad=False)

            if FLAGS.comb_mask:
                steps = tf.constant(0)
                c = lambda i, x: tf.less(i, FLAGS.num_steps)

                def langevin_attention_step(counter, attention_mask):
                    attention_mask = attention_mask + tf.random_normal(tf.shape(attention_mask), mean=0.0, stddev=0.01)
                    energy_noise = model.forward(
                        x_mod1,
                        x_mod2,
                        weights,
                        attention_mask,
                        label=LABEL_SPLIT[j],
                        reuse=True,
                        stop_at_grad=False,
                        stop_batch=True)

                    if FLAGS.heir_mask:
                        raise NotImplementedError

                    attention_grad = tf.gradients(
                        FLAGS.temperature * energy_noise, [attention_mask])[0]
                    energy_noise_old = energy_noise

                    # Clip gradient norm for now
                    attention_mask = attention_mask - (FLAGS.attention_lr) * attention_grad
                    attention_mask = tf.layers.average_pooling2d(attention_mask, (3, 3), 1, padding='SAME')
                    attention_mask = tf.stop_gradient(attention_mask)

                    counter = counter + 1

                    return counter, attention_mask

                steps, attention_mask = tf.while_loop(c, langevin_attention_step, (steps, attention_mask))

                # attention_mask = tf.Print(attention_mask, [attention_mask])

                energy_pos = model.forward(
                    X_SPLIT1[j],
                    X_SPLIT2[j],
                    weights,
                    tf.stop_gradient(attention_mask),
                    label=LABEL_POS_SPLIT[j],
                    stop_at_grad=False)

                if FLAGS.heir_mask:
                    raise NotImplementedError

            else:
                # positive sample energy
                energy_pos = model.forward(
                    X_SPLIT1[j],
                    X_SPLIT2[j],
                    weights,
                    attention_mask,
                    label=LABEL_POS_SPLIT[j],
                    stop_at_grad=False)

                if FLAGS.heir_mask:
                    raise NotImplementedError

            print("Building graph...")
            x_mod1 = X_NOISE_SPLIT1[j]
            x_mod2 = X_NOISE_SPLIT2[j]

            x_grads = []

            eps_begin = tf.zeros(1)

            steps = tf.constant(0)
            c_cond = lambda i, x1, x2, y: tf.less(i, FLAGS.num_steps)

            def langevin_step(counter, x_mod1, x_mod2, attention_mask):
                lr = FLAGS.step_lr

                x_mod1 = x_mod1 + tf.random_normal(tf.shape(x_mod1),
                                                   mean=0.0,
                                                   stddev=0.001 * FLAGS.rescale * FLAGS.noise_scale)
                x_mod2 = x_mod2 + tf.random_normal(tf.shape(x_mod2),
                                                   mean=0.0,
                                                   stddev=0.001 * FLAGS.rescale * FLAGS.noise_scale)
                attention_mask = attention_mask + tf.random_normal(tf.shape(attention_mask), mean=0.0, stddev=0.01)

                energy_noise = model.forward(
                    x_mod1,
                    x_mod2,
                    weights,
                    attention_mask,
                    label=LABEL_SPLIT[j],
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True)

                x_grad1, x_grad2, attention_grad = tf.gradients(
                    FLAGS.temperature * energy_noise, [x_mod1, x_mod2, attention_mask])

                if not FLAGS.comb_mask:
                    attention_grad = tf.zeros(1)

                if FLAGS.proj_norm != 0.0:
                    if FLAGS.proj_norm_type == 'l2':
                        x_grad1 = tf.clip_by_norm(x_grad1, FLAGS.proj_norm)
                        x_grad2 = tf.clip_by_norm(x_grad2, FLAGS.proj_norm)
                    elif FLAGS.proj_norm_type == 'li':
                        x_grad1 = tf.clip_by_value(
                            x_grad1, -FLAGS.proj_norm, FLAGS.proj_norm)
                        x_grad2 = tf.clip_by_value(
                            x_grad2, -FLAGS.proj_norm, FLAGS.proj_norm)
                    else:
                        print("Other types of projection are not supported!!!")
                        assert False

                # Clip gradient norm for now
                x_last1 = x_mod1 - (lr) * x_grad1
                x_last2 = x_mod2 - (lr) * x_grad2

                if FLAGS.comb_mask:
                    attention_mask = attention_mask - FLAGS.attention_lr * attention_grad
                    attention_mask = tf.layers.average_pooling2d(attention_mask, (3, 3), 1, padding='SAME')
                    attention_mask = tf.stop_gradient(attention_mask)

                x_mod1 = x_last1
                x_mod2 = x_last2
                x_mod1 = tf.clip_by_value(x_mod1, 0, FLAGS.rescale)
                x_mod2 = tf.clip_by_value(x_mod2, 0, FLAGS.rescale)

                counter = counter + 1

                return counter, x_mod1, x_mod2, attention_mask

            # sequentially refine negative samples
            steps, x_mod1, x_mod2, attention_mask = tf.while_loop(c_cond, langevin_step,
                                                                  (steps, x_mod1, x_mod2, attention_mask))

            attention_mask = tf.stop_gradient(attention_mask)

            # negative sample energy
            energy_eval = model.forward(
                x_mod1,
                x_mod2,
                weights,
                attention_mask,
                label=LABEL_SPLIT[j],
                stop_at_grad=False,
                reuse=True)
            x_grad1, x_grad2, attention_grad = tf.gradients(FLAGS.temperature * energy_eval,
                                                            [x_mod1, x_mod2, attention_mask])
            x_grads.append((x_grad1, x_grad2))

            energy_neg = model.forward(
                tf.stop_gradient(x_mod1),
                tf.stop_gradient(x_mod2),
                weights,
                tf.stop_gradient(attention_mask),
                label=LABEL_SPLIT[j],
                stop_at_grad=False,
                reuse=True)

            temp = FLAGS.temperature

            x_off1 = tf.reduce_mean(
                tf.abs(x_mod1[:tf.shape(X1)[0]] - X1))
            x_off2 = tf.reduce_mean(
                tf.abs(x_mod2[:tf.shape(X2)[0]] - X2))

            loss_energy = model.forward(
                x_mod1,
                x_mod2,
                weights,
                attention_mask,
                reuse=True,
                label=LABEL_SPLIT[j],
                stop_grad=True)

            print("Finished processing loop construction ...")

            test_x_mod1 = x_mod1
            test_x_mod2 = x_mod2

            if FLAGS.cclass or FLAGS.model_cclass:
                label_sum = tf.reduce_sum(LABEL_SPLIT[j], axis=0)
                label_prob = label_sum / tf.reduce_sum(label_sum)
                label_ent = -tf.reduce_sum(label_prob *
                                           tf.math.log(label_prob + 1e-7))
            else:
                label_ent = tf.zeros(1)

            target_vars['label_ent'] = label_ent

            if FLAGS.train:
                if FLAGS.objective == 'logsumexp':
                    energy_neg_reduced = (energy_neg - tf.reduce_min(energy_neg))
                    coeff = tf.stop_gradient(tf.exp(-temp * energy_neg_reduced))
                    norm_constant = tf.stop_gradient(tf.reduce_sum(coeff)) + 1e-4
                    pos_loss = tf.reduce_mean(temp * energy_pos)
                    neg_loss = coeff * (-1 * temp * energy_neg) / norm_constant
                    loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
                elif FLAGS.objective == 'cd':
                    pos_loss = tf.reduce_mean(temp * energy_pos)
                    neg_loss = -tf.reduce_mean(temp * energy_neg)
                    loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
                elif FLAGS.objective == 'softplus':
                    loss_ml = FLAGS.ml_coeff * \
                              tf.nn.softplus(temp * (energy_pos - energy_neg))
                else:
                    raise RuntimeError

                loss_total = tf.reduce_mean(loss_ml)

                if not FLAGS.zero_kl:
                    loss_total = loss_total + tf.reduce_mean(loss_energy)

                loss_total = loss_total + \
                             FLAGS.l2_coeff * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(
                    tf.square((energy_neg))))

                print("Started gradient computation...")
                gvs = optimizer.compute_gradients(loss_total)
                gvs = [(k, v) for (k, v) in gvs if k is not None]

                print("Applying gradients...")

                tower_grads.append(gvs)

                print("Finished applying gradients.")

                target_vars['loss_ml'] = loss_ml
                target_vars['total_loss'] = loss_total
                target_vars['loss_energy'] = loss_energy
                target_vars['weights'] = weights
                target_vars['gvs'] = gvs

            target_vars['X1'] = X1
            target_vars['X2'] = X2
            target_vars['Z'] = Z
            target_vars['LABEL'] = LABEL
            target_vars['HIER_LABEL'] = HEIR_LABEL
            target_vars['LABEL_POS'] = LABEL_POS
            target_vars['X_NOISE1'] = X_NOISE1
            target_vars['X_NOISE2'] = X_NOISE2
            target_vars['energy_pos'] = energy_pos
            target_vars['attention_grad'] = attention_grad

            if len(x_grads) >= 1:
                target_vars['x_grad'] = x_grads[-1]
                target_vars['x_grad_first'] = x_grads[0]
            else:
                target_vars['x_grad'] = tf.zeros(1)
                target_vars['x_grad_first'] = tf.zeros(1)

            target_vars['x_mod1'] = x_mod1
            target_vars['x_mod2'] = x_mod2
            target_vars['x_off1'] = x_off1
            target_vars['x_off2'] = x_off2
            target_vars['temp'] = temp
            target_vars['energy_neg'] = energy_neg
            target_vars['test_x_mod1'] = test_x_mod1
            target_vars['test_x_mod2'] = test_x_mod2
            target_vars['eps_begin'] = eps_begin
            target_vars['ATTENTION_MASK'] = ATTENTION_MASK
            target_vars['models_pretrain'] = None  # models_pretrain
            target_vars['Z1'] = Z1
            target_vars['Z2'] = Z2
            target_vars['energy_z'] = energy_z
            if FLAGS.comb_mask:
                target_vars['attention_mask'] = tf.nn.softmax(attention_mask)
            else:
                target_vars['attention_mask'] = tf.zeros(1)

        if FLAGS.train:
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads)
            target_vars['train_op'] = train_op

    # sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=4)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))

    sess.run(tf.global_variables_initializer())

    resume_itr = 0

    if (FLAGS.resume_iter != -1 or not FLAGS.train):
        model_file = os.path.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter + 1
        logger.step = FLAGS.resume_iter // FLAGS.log_interval + 1
        optimistic_restore(sess, model_file)

    print("Initializing variables...")

    print("Start broadcast")
    print("End broadcast")

    test(target_vars, saver, sess, logger, test_dataset)


if __name__ == "__main__":
    main()
