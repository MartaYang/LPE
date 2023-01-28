import os
import wandb
import torch
import pprint
import random
import argparse
import numpy as np
from termcolor import colored


def setup_run(arg_mode='train'):
    args = parse_args(arg_mode=arg_mode)
    pprint(vars(args))

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'checkpoints/{args.dataset}/{args.shot}shot-{args.way}way/', args.extra_dir)
    ensure_path(args.save_path)

    if not args.no_wandb:
        wandb.init(project=f'LPE-{args.dataset}-{args.way}w{args.shot}s',
                   config=args,
                   save_code=True,
                   mode="offline" if args.wandb_offline else "online",
                   name=args.extra_dir)
        wandb.run.log_code(".")

    if args.dataset == 'miniimagenet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
    elif args.dataset == 'cifar_fs':
        args.num_class = 64

    return args


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def load_model(model, dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(dir)['params']
    # print(pretrained_dict.keys())
    # print('\n')
    # print(model_dict.keys())

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:
        print('loading model from :', dir)
        if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
            if 'module' in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
        model.load_state_dict(model_dict)

    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def detect_grad_nan(model):
    for param in model.parameters():
        if param.grad is None: # TODO
            continue            #TODO!!!
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()


def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow


def parse_args(arg_mode):
    parser = argparse.ArgumentParser(description='TODO')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='miniimagenet',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'cifar_fs', 'awa2'])
    parser.add_argument('-data_dir', type=str, default='/data/FSLDatasets/LPE_dataset', help='dir of datasets')

    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=128, help='auxiliary batch size')
    parser.add_argument('-temperature', type=float, default=0.1, metavar='tau', help='temperature for metric-based loss')
    parser.add_argument('-lamb', type=float, default=0.25, metavar='lambda', help='loss balancing term for few-shot classification loss')
    parser.add_argument('-lamb_diff', type=float, default=1, metavar='lamb_diff', help='loss balancing term for divergent loss')

    ''' about training schedules '''
    parser.add_argument('-max_epoch', type=int, default=80, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70], help='milestones for MultiStepLR')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')

    ''' about env '''
    parser.add_argument('-gpu', default='0', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-extra_dir', type=str, default='test222', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-no_wandb', action='store_true', help='not plotting learning curve on wandb',
                        default=arg_mode == 'test')  # train: enable logging / test: disable logging
    parser.add_argument('-wandb_offline', action='store_true', default=False, help='wandb_offline')

    ''' about LPE (our Latent Parts Embedding)'''
    parser.add_argument('-is_LPE', action='store_true', default=False, help='using Semantic Guided Latent Parts Embedding')
    parser.add_argument('-semantic_path', type=str, default=None, help='the semantic embeddings for all concepts corresponding to current dataset')
    parser.add_argument('-sem_dim', type=int, default=300, help='the dimension of current using semantic embedding (e.g., word embedding is 300 dim)')
    parser.add_argument('-n_attr_templet', type=int, default=10, help='number of latent parts P')
    parser.add_argument('-templet_weight', type=str, default='sem_generate', help='ways to generate the weights for each part (the weights are used for weighted parts scores fusion)')
    parser.add_argument('-no_transferbase', action='store_true', default=False, help='whether or not transfer part-level prior from base classes')
    parser.add_argument('-no_queryatt', action='store_true', default=False, help='whether or not apply the latent parts embedding to the query samples')
    parser.add_argument('-lambda_sqr', action='store_true', default=False, help='whether or not square the tmpweight_lambda when calculating the weights_on_templet')
    parser.add_argument('-shotscore_strategy', type=str, default='mean', help='how to deal with the multi-shots')
    

    args = parser.parse_args()
    return args
