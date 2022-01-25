In Defence of the Learning Without Forgetting for Task Incremental Learning
===========================================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

Introduction
------------

This repository includes the code to reproduce the results presented in the paper. The repository contains two codebases,
hat, and hypercl. The first, which is based on the official GitHub repository of HAT: https://github.com/joansj/hat ,
is used to reproduce results for the methods: LwF, EWC, IMM-MEAN, IMM-MODE, HAT, and JOINT. The later, which based on
the official GitHub repository of HyperCL: https://github.com/chrhenning/hypercl , is used to reproduce results for
HyerCL methods.

Python requirements
-------------------

The code was tested under PyTorch 1.1 and the python package requirements are presented in requirements.txt.

Data requirements
-----------------

CIFAR-100
~~~~~~~~~

The data will be automatically downloaded in both codebases, no further action is required.

Tiny-ImageNet
~~~~~~~~~~~~~

run.sh should be executed within the  prepare_tiny_imagenet folder. This will create the folder tiny-imagenet-200,
as well a soft link inside the hat codebase.

HAT codebase
-------------

In order to run a single setting in this codebase, one should run inside ``hat/src`` folder, for example:

.. code-block:: console

    $ python run.py --experiment only_cifar100 --approach lwf --arch wide-resnet20-w5 --aug aug-v1 --nepochs 200 --task_size 5 --lr 0.01 --parameter 1,2


To see the full set of available options run

.. code-block:: console

    $ python run.py --help

HyperCL codebase
-----------------

This codebase is further split into two folders: ``cifar`` and ``tiny_imagenet``. Each contains the code
necessary to run an experiment for one of the datasets. For instance, in order to run a single setting for
CIFAR experiment, one should run inside ``hypercl/cifar`` folder:

.. code-block:: console

    $ python train_resnet.py --use_adam --custom_network_init --plateau_lr_scheduler --lambda_lr_scheduler --shuffle_order --net_type resnet32

To see the full set of available options run in either of the folders:

.. code-block:: console

    $ python train_resnet.py --help

General notes
-------------

* while the code in ``hypercl/cifar`` can automatically find the data path for CIFAR, it is important to use the flag ``--data_dir`` for TinyImageNet experiments with the root path to the TinyImageNet dataset produced in `Data requirements`.

* Don`t run CIFAR or TinyImageNet experiments for the first time with the same seed in parallel (no matter which method). Since the seed is used also to create the task split and each codebase uses a folder as a cache to save splits for each previously seen seeds. After a given seed is run in each codebase separately, it is ok (and recommended if possible) to run other experiments with this seed in parallel for other methods.

* ``random_seeds.txt`` include the 5 random seeds used to produce the paper results.

Reproduce of table 1 results
----------------------------

``table_1_experiments.sh`` contains all runs to reproduce the table 1 results presented in the paper, single line per
run. It is necessary to run these commands inside ``hat/src`` folder. It is possible to run all different commands in
parallel in the following way (influenced by the above notes):

1. run the 20 first commands (in parallel or not)
2. run the other commands (in parallel or not)

The raw results will be saved in ``hat/res``. In order to extract them, run:

.. code-block:: console

    $ python hat/src/extract_results.py --res_dir hat/res --data cifar --n_try 5

.. code-block:: console

    $ python hat/src/extract_results.py --res_dir hat/res --data tiny_imagenet --n_try 5

Reproduce of table 2 results
----------------------------

``table_2_experiments.sh`` contains all runs to reproduce the table 2 results presented in the paper, single line per
run. It is necessary to run these commands inside ``hat/src`` folder. It is possible to run all different commands in
parallel in the following way (influenced by the above notes):

1. run the 20 first commands (in parallel or not)
2. run the other commands (in parallel or not)

The raw results will be saved in ``hat/res``. In order to extract them, run:

.. code-block:: console

    $ python hat/src/extract_results.py --res_dir hat/res --data cifar --n_try 5

.. code-block:: console

    $ python hat/src/extract_results.py --res_dir hat/res --data tiny_imagenet --n_try 5

We would like to note that the 20 first runs overlap with table 1 results above (first 20 commands in each ``.sh`` file),
so if one already executed these commands for table 1, it is possible to skip the first 20 commands and run in
parallel all others.

Reproduce of table 3 results
----------------------------

* ``table_3_experiments_hat.sh`` contains all runs to reproduce the table 3 results presented in the paper for LwF, HAT, EWC, IMM-MEAN, IMM-MODE, and JOINT. It is necessary to run these commands inside ``hat/src`` folder.
* ``table_3_experiments_hypercl_cifar.sh`` contains all runs to reproduce the table 3 results presented in the paper for **CIFAR** with HyperCL, and should be executed under ``hypercl/cifar``.
* ``table_3_experiments_hypercl_tiny_imagenet.sh`` contains all runs to reproduce the table 3 results presented in the paper for **TinyImageNet** with HyperCL, and should be executed under ``hypercl/tiny_imagenet``.

It is possible to run all different commands in parallel following similar guidelines presented in the notes and in
the previous sections. Moreover, 40 first commands in ``table_3_experiments_hat.sh`` are essentially the same commands
from the previous section and can be skipped if already run.

The raw results for ``table_3_experiments_hat.sh`` will be saved in ``hat/res``. In order to extract them, run:

.. code-block:: console

    $ python hat/src/extract_results.py --res_dir hat/res --data cifar --n_try 5

.. code-block:: console

    $ python hat/src/extract_results.py --res_dir hat/res --data tiny_imagenet --n_try 5

The raw results for ``table_3_experiments_hypercl_cifar.sh`` will be saved in ``hypercl/cifar/out_resnet``.
In order to extract them, run:

.. code-block:: console

    $ python hypercl/cifar/extract_results.py --res_dir hypercl/cifar/out_resnet --n_try 5

The raw results for ``table_3_experiments_hypercl_tiny_imagenet.sh`` will be saved in ``hypercl/tiny_imagenet/out_resnet``.
In order to extract them, run:

.. code-block:: console

    $ python hypercl/tiny_imagenet/extract_results.py --res_dir hypercl/tiny_imagenet/out_resnet --n_try 5




