import random
import string
import sys, os, argparse, time
import numpy as np
import torch
from torchvision import transforms

import utils

tstart = time.time()

# Arguments
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='', type=str, required=True, choices=['only_cifar100', 'tiny_imagenet'],
                    help='(default=%(default)s)')
parser.add_argument('--approach', default='', type=str, required=True, choices=['lwf', 'ewc', 'imm-mean', 'imm-mode',
                                                                                'joint', 'hat'],
                    help='(default=%(default)s)')
parser.add_argument('--arch', default='', type=str, required=True, choices=['alexnet',
                                                                            'alexnet-no-drop',
                                                                            'resnet32',
                                                                            'resnet20',
                                                                            'resnet62',
                                                                            'wide-resnet20-w2',
                                                                            'wide-resnet20-w5',
                                                                            'wide-resnet20-w8'],
                    help='(default=%(default)s)')
parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--nepochs', default=200, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--task_size', default=20, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--aug', default='aug-v1', type=str, choices=['aug-v1', 'aug-v2'], help='(default=%(default)s)')
parser.add_argument('--disable_data_augment', action='store_true')

args = parser.parse_args()

use_data_augment = not args.disable_data_augment

if args.output == '':
    random_postfix = ''.join(random.choices(string.ascii_uppercase +
                                            string.digits, k=10))
    args.output = '../res/' + args.experiment + '_' + args.approach + '-' + args.arch + '_' + str(args.seed) + '_' + \
                  (args.aug if use_data_augment else "no_augment") + "_" + \
                  time.strftime("%Y-%m-%d__%H-%M-%S") + "_" + random_postfix + '.txt'

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment == 'only_cifar100':
    from dataloaders import only_cifar100 as dataloader
elif args.experiment == 'tiny_imagenet':
    from dataloaders import tiny_imagenet as dataloader
else:
    raise NotImplementedError

# Args -- Approach
if args.approach == 'lwf':
    from approaches import lwf as approach
elif args.approach == 'ewc':
    from approaches import ewc as approach
elif args.approach == 'imm-mean':
    from approaches import imm_mean as approach
elif args.approach == 'imm-mode':
    from approaches import imm_mode as approach
elif args.approach == 'hat':
    from approaches import hat as approach
elif args.approach == 'joint':
    from approaches import joint as approach
else:
    raise NotImplementedError

# Args -- Network
if args.approach == 'hat':
    if args.arch == 'alexnet':
        from networks import alexnet_hat as network
    elif args.arch == "wide-resnet20-w8":
        from networks import wide_resnet_w_8_hat_no_down_gating as network
    elif args.arch == "wide-resnet20-w8-bn":
        from networks import wide_resnet_w_8_hat_no_down_gating_bn as network
    else:
        raise NotImplementedError
else:
    if args.arch == "alexnet":
        from networks import alexnet as network
    elif args.arch == "alexnet-no-drop":
        from networks import alexnet_no_drop as network
    elif args.arch == 'resnet32':
        from networks import resnet32 as network
    elif args.arch == 'resnet62':
        from networks import resnet62 as network
    elif args.arch == 'resnet20':
        from networks import resnet20 as network
    elif args.arch == 'wide-resnet20-w2':
        from networks import wide_resnet20_w_2 as network
    elif args.arch == 'wide-resnet20-w5':
        from networks import wide_resnet20_w_5 as network
    elif args.arch == 'wide-resnet20-w8':
        from networks import wide_resnet20_w_8 as network
    else:
        raise NotImplementedError

########################################################################################################################

# Load
print('Load data...')
if args.experiment in ['only_cifar100', 'tiny_imagenet']:
    kwargs = {"task_size": args.task_size}
else:
    kwargs = {}
data, taskcla, inputsize = dataloader.get(seed=args.seed, **kwargs)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(inputsize, taskcla).cuda()

utils.print_model_report(net)

if use_data_augment:
    if args.aug == 'aug-v2':
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
    elif args.aug == 'aug-v1':
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

    if args.experiment in ['tiny_imagenet']:
        transform = transforms.Compose(
            [
                transforms.Normalize((-1, -1, -1), (2, 2, 2)),
                transforms.ToPILImage('RGB'),
                augment,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    elif args.experiment in ['only_cifar100']:

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        transform = transforms.Compose(
            [
                transforms.Normalize([-m / s for m, s in zip(mean, std)], [1. / s for s in std]),
                transforms.ToPILImage('RGB'),
                augment,
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
    else:
        raise NotImplementedError
else:
    transform = None

appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args, transform=transform)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    if args.experiment in ['only_cifar100', 'tiny_imagenet']:
        print('Task {:2d} classes: {}'.format(t, data[t]['classes']))
    print('*' * 100)

    if args.approach in ['joint']:
        # Get data. We do not put it to GPU
        if t == 0:
            xtrain = data[t]['train']['x']
            ytrain = data[t]['train']['y']
            xvalid = data[t]['valid']['x']
            yvalid = data[t]['valid']['y']
            task_t = t * torch.ones(xtrain.size(0)).int()
            task_v = t * torch.ones(xvalid.size(0)).int()
            task = [task_t, task_v]
        else:
            xtrain = torch.cat((xtrain, data[t]['train']['x']))
            ytrain = torch.cat((ytrain, data[t]['train']['y']))
            xvalid = torch.cat((xvalid, data[t]['valid']['x']))
            yvalid = torch.cat((yvalid, data[t]['valid']['y']))
            task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
            task = [task_t, task_v]
    else:
        # Get data
        xtrain = data[t]['train']['x'].cuda()
        ytrain = data[t]['train']['y'].cuda()
        xvalid = data[t]['valid']['x'].cuda()
        yvalid = data[t]['valid']['y'].cuda()
        task = t

    # Train
    appr.train(task, xtrain, ytrain, xvalid, yvalid)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')

# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        # save task names
        from copy import deepcopy

        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t, ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t] = deepcopy(acc[t, :])
            appr.logs['test_loss'][t] = deepcopy(lss[t, :])
        # pickle
        import gzip
        import pickle

        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
