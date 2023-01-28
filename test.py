import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.LPE import LPE

import numpy as np

def evaluate(epoch, model, loader, args=None, set='val', label2vec=None):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, test_labels) in enumerate(tqdm_gen, 1):
            data, test_labels = data.cuda(), test_labels.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]

            if args.is_LPE:
                model.module.mode = 'LPE'
                support_labels = test_labels[:args.way]
                logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query,\
                                                 label2vec[support_labels], support_labels, label[:k]))
            else:
                model.module.mode = 'plain_cosine'
                logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))

            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test', label2vec = test_set.label2vec)
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = LPE(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
