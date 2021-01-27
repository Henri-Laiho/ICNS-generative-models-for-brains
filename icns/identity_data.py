import os
import random

import pandas as pd
import numpy as np
from tensorflow.python.platform import flags
from torch.utils.data import Dataset
from scipy.misc import imread, imresize

FLAGS = flags.FLAGS


class CelebAPairs(Dataset):

    def __init__(self, samples_per_ground=None, cycles_per_side=2, minimum_occurrences=5,
                 random_state=None, pos_probability=0.5):
        if samples_per_ground is None:
            samples_per_ground = FLAGS.batch_size * 3
        self.rand = random.Random(random_state)
        self.path = os.path.join("CelebA", "img_align_celeba")
        self.ident = pd.read_csv("CelebA/Anno/identity_CelebA.txt", sep="\s+", names=['file', 'celeb_id'])
        self.samples_per_ground = samples_per_ground
        self.cycles_per_side = cycles_per_side
        self.pos_probability = pos_probability
        self.side_state = None
        self.cycle = 0
        self.side = 0
        self.sample_counter = 0

        ivc = self.ident.celeb_id.value_counts()
        self.ivcthr = ivc[ivc >= minimum_occurrences]

        imgs_of_celeb = {}
        for x in self.ivcthr.index:
            imgs_of_celeb[x] = self.ident[self.ident.celeb_id == x].file
        self.imgs_of_celeb = imgs_of_celeb

        self.fnames = pd.Series()
        img_to_celeb = {}
        for x in imgs_of_celeb:
            labels = [y for y in imgs_of_celeb[x]]
            for label in labels:
                img_to_celeb[label] = x
            self.fnames = self.fnames.append(imgs_of_celeb[x])
        self.img_to_celeb = img_to_celeb
        self.fnames = self.fnames.reset_index(drop=True)

    def __len__(self):
        return self.fnames.shape[0]

    def __getitem__(self, index):
        if FLAGS.single:
            index = 0

        label = self.rand.choices([0, 1], weights=[1 - self.pos_probability, self.pos_probability])[0]
        if label:  # positive
            if self.side_state is not None:
                side_image, side_fname, celeb = self.side_state
                main_fname = self.imgs_of_celeb[celeb].iloc[index % len(self.imgs_of_celeb[celeb])]
            else:
                main_fname = self.fnames[index]
                celeb = self.get_id_for_fname(main_fname)
                side_image = None
                side_fname = self.imgs_of_celeb[celeb].iloc[index % len(self.imgs_of_celeb[celeb])]
        else:  # negative
            if self.side_state is not None:
                side_image, side_fname, celeb = self.side_state
                while self.get_id_for_fname(self.fnames[index % len(self)]) == celeb:
                    index += len(self.imgs_of_celeb)
                main_fname = self.fnames[index % len(self)]
            else:
                main_fname = self.fnames[index]
                celeb = self.get_id_for_fname(main_fname)
                side_image = None
                side_fname = self.rand.choice(self.fnames)
                while self.get_id_for_fname(side_fname) == celeb:
                    side_fname = self.rand.choice(self.fnames)

        im_main, image_size = self.load_im(main_fname)
        if side_image is None:
            im_side, image_size2 = self.load_im(side_fname)
            assert image_size == image_size2
        else:
            im_side = side_image

        self.side_state = side_image, side_fname, celeb

        im1, im2 = (im_side, im_main) if self.side else (im_main, im_side)
        if FLAGS.datasource == 'default':
            im_corrupt1 = im1 + 0.3 * np.random.randn(image_size, image_size, 3)
            im_corrupt2 = im2 + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt1 = np.random.uniform(
                0, 1, size=(image_size, image_size, 3))
            im_corrupt2 = np.random.uniform(
                0, 1, size=(image_size, image_size, 3))
        else:
            raise RuntimeError('invalid datasource')

        self.sample_counter += 1
        if self.sample_counter % self.samples_per_ground == 0:
            self.side_state = None
            self.cycle += 1
            if self.cycle % self.cycles_per_side == 0:
                self.side = not self.side

        label = np.eye(2)[label]
        return (im_corrupt1, im_corrupt2), (im1, im2), label

    def get_id_for_fname(self, fname):
        return self.img_to_celeb[fname]

    def load_im(self, fname):
        path = os.path.join(self.path, fname)
        im = imread(path)
        im = imresize(im, (128, 128))
        image_size = 128
        im = im / 255.
        return im, image_size
