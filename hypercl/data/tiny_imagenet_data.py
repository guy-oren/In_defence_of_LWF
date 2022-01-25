#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title           :data/cifar100_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :05/02/2019
# @version         :1.0
# @python_version  :3.6.8
"""
TinyImagenet Dataset
-----------------

The module :mod:`data.cifar100_data` contains a handler for the CIFAR 100
dataset.

The dataset consists of 60000 32x32 colour images in 100 classes, with 600
images per class. There are 50000 training images and 10000 test images.

Information about the dataset can be retrieved from:
    https://www.cs.toronto.edu/~kriz/cifar.html
"""
# FIXME: The content of this module is mostly a copy of the module
# 'cifar10_data'. These two should be merged in future.

import os
import numpy as np
import time
import _pickle as pickle
import urllib.request
import tarfile
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

from data.dataset import Dataset
from data.cifar10_data import CIFAR10Data

class TinyImagenetData(Dataset):
    """An instance of the class shall represent the TinyImagenet dataset.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.
        use_data_augmentation (bool, optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor` (hence,
            **only available for PyTorch**, so far).
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
    """

    _TRAIN_BATCH_FN = 'train'
    _TEST_BATCH_FN = 'val'

    def __init__(self, data_path, use_one_hot=False,
                 use_data_augmentation=False, aug_ver='aug-v1', validation_size=5000):
        super().__init__()

        start = time.time()

        print('Reading TinyImagenet dataset ...')

        assert os.path.exists(data_path)

        train_batch_fn = os.path.join(data_path,
                                     TinyImagenetData._TRAIN_BATCH_FN)
        test_batch_fn = os.path.join(data_path,
                                     TinyImagenetData._TEST_BATCH_FN)

        assert(os.path.exists(train_batch_fn) and
               os.path.exists(test_batch_fn))

        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['num_classes'] = 200
        self._data['is_one_hot'] = use_one_hot
        
        self._data['in_shape'] = [32, 32, 3]
        self._data['out_shape'] = [200 if use_one_hot else 1]

        self._read_batches(train_batch_fn, test_batch_fn, validation_size)

        # Initialize PyTorch data augmentation.
        self._augment_inputs = False
        if use_data_augmentation:
            self._augment_inputs = True
            self._train_transform, self._test_transform = \
                self.torch_input_transforms(aug_ver)
    
        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    @staticmethod
    def torch_input_transforms(aug_ver='aug-v1'):
        """Get data augmentation pipelines for CIFAR-10 inputs.

        Note, the augmentation is inspired by the augmentation proposed in:
            https://www.aiworkbox.com/lessons/augment-the-cifar10-dataset-using\
-the-randomhorizontalflip-and-randomcrop-transforms

        Note:
            We use the same data augmentation pipeline for CIFAR-100, as the
            images are very similar. Here is an example where they use slightly
            different normalization values, but we ignore this for now:
            https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html

        Returns:
            (tuple): Tuple containing:

                - **train_transform**: A transforms pipeline that applies random
                  transformations and normalizes the image.
                - **test_transform**: Similar to train_transform, but no random
                  transformations are applied.
        """
        # Copyright 2017-2018 aiworkbox.com
        # Unfortunately, no license was visibly provided with this code.
        # Though, we note that the original license applies regarding the parts
        # of the code that have been copied from the above mentioned website (we
        # slightly modified this code).
        #
        # Note, that we use this code WITHOUT ANY WARRANTIES.

        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        if aug_ver == 'aug-v2':
            augment = transforms.RandomApply(
                [
                    transforms.Compose(
                        [
                            transforms.Pad(
                                int(np.ceil(0.1 * 32)), padding_mode="reflect"
                            ),
                            transforms.RandomAffine(0, translate=(0.2, 0.2)),
                            transforms.CenterCrop(32),
                        ]
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                ], p=0.6)
        elif aug_ver == 'aug-v1':
            augment = transforms.RandomApply(
                [
                    transforms.Compose(
                        [
                            transforms.Pad(
                                int(np.ceil(0.1 * 32)), padding_mode="reflect"
                            ),
                            transforms.RandomAffine(0, translate=(0.1, 0.1)),
                            transforms.CenterCrop(32),
                        ]
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
                ], p=0.6)
        else:
            raise NotImplementedError

        train_transform = transforms.Compose([
            transforms.ToPILImage('RGB'),
            augment,
            transforms.ToTensor(),
            # TODO: could we have problem with normalize because of the color jitter?
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage('RGB'),
            transforms.ToTensor(),
            normalize,
        ])

        return train_transform, test_transform

    def _read_batches(self, train_fn, test_fn, validation_size):
        """Read training and testing batch from files.

        The method fills the remaining mandatory fields of the _data attribute,
        that have not been set yet in the constructor.

        The images are converted to match the output shape (32, 32, 3) and
        scaled to have values between 0 and 1. For labels, the correct encoding
        is enforced.

        Args:
            train_fn: Filepath of the train batch.
            test_fn: Filepath of the test batch.
            validation_size: Number of validation samples.
        """
        # Read test batch.
        test_dadataset = ImageFolder(test_fn)
        test_samples = []
        test_labels = []
        for i in range(len(test_dadataset)):
            img, label = test_dadataset[i]
            test_samples.append(np.array(img))
            test_labels.append(label)

        test_samples = np.stack(test_samples)
        test_labels = np.stack(test_labels)

        # Read train batch.
        train_dadataset = ImageFolder(train_fn)
        train_samples = []
        train_labels = []
        for i in range(len(train_dadataset)):
            img, label = train_dadataset[i]
            train_samples.append(np.array(img))
            train_labels.append(label)

        train_samples = np.stack(train_samples)
        train_labels = np.stack(train_labels)

        if validation_size > 0:
            assert(validation_size < train_labels.shape[0])
            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_labels.size)

        else:
            train_inds = np.arange(train_labels.size)

        test_inds = np.arange(train_labels.size, 
                              train_labels.size + test_labels.size)

        labels = np.concatenate([train_labels, test_labels])
        labels = np.reshape(labels, (-1, 1))

        images = np.concatenate([train_samples, test_samples], axis=0)

        # Note, images are currently encoded in a way, that there shape
        # corresponds to (3, 32, 32). For consistency reasons, we would like to
        # change that to (32, 32, 3).
        images = np.reshape(images, (-1, 3, 32, 32))
        images = np.rollaxis(images, 1, 4)
        images = np.reshape(images, (-1, 32 * 32 * 3))
        # Scale images into a range between 0 and 1.
        images = images / 255

        self._data['in_data'] = images
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds
        if validation_size > 0:
            self._data['val_inds'] = val_inds

        if self._data['is_one_hot']:
            labels = self._to_one_hot(labels)

        self._data['out_data'] = labels

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Tiny Imagenet'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, this method has been overwritten from the base class.

        The input images are preprocessed if data augmentation is enabled.
        Preprocessing involves normalization and (for training mode) random
        perturbations.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if self._augment_inputs and not force_no_preprocessing:
            if mode == 'inference':
                transform = self._test_transform
            elif mode == 'train':
                transform = self._train_transform
            else:
                raise ValueError('"%s" not a valid value for argument "mode".'
                                 % mode)

            return CIFAR10Data.torch_augment_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(self, x, device,
                mode=mode, force_no_preprocessing=force_no_preprocessing)
        
    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Re-Implementation of method
        :meth:`data.dataset.Dataset._plot_config`.

        This method has been overriden to ensure, that there are 2 subplots,
        in case the predictions are given.
        """
        plot_configs = super()._plot_config(inputs, outputs=outputs,
                                            predictions=predictions)
        
        if predictions is not None and \
                np.shape(predictions)[1] == self.num_classes:
            plot_configs['outer_hspace'] = 0.6
            plot_configs['inner_hspace'] = 0.4
            plot_configs['num_inner_rows'] = 2
            #plot_configs['num_inner_cols'] = 1
            plot_configs['num_inner_plots'] = 2

        return plot_configs

if __name__ == '__main__':
    pass


