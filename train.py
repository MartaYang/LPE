import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run, load_model
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.LPE import LPE
from test import test_main, evaluate

def train(epoch, model, loader, optimizer, args=None, label2vec=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    loss_diff_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), data_label_aux) in enumerate(zip(tqdm_gen, train_loader_aux), 1):
        (data_aux, train_labels_aux) = data_label_aux

        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux) 

        # loss for batch
        if args.is_LPE:
            model.module.mode = 'LPE'
            data_shot, data_query = data[:k], data[k:]
            support_labels = train_labels[:args.way]
            logits, absolute_logits, loss_diff = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query,\
                                                        label2vec[support_labels], support_labels, label[:k]))
        else:
            model.module.mode = 'plain_cosine'
            data_shot, data_query = data[:k], data[k:]
            logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
            loss_diff = torch.tensor(0.0)
        epi_loss = F.cross_entropy(logits, label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss

        # loss for auxiliary batch (under semantic filters)
        if args.is_LPE:
            model.module.mode = 'fc_sem_filters'
            logits_aux = model((data_aux, label2vec))
            loss_sem_fc = F.cross_entropy(logits_aux, train_labels_aux)
            loss_aux += loss_sem_fc

        loss = args.lamb * (epi_loss) + args.lamb_diff * loss_diff + loss_aux
        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        loss_diff_meter.update(loss_diff.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f}({loss_diff_meter.avg():.4f}) | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=False)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=False) #False to avoid Warning

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=False)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    set_seed(args.seed)
    model = LPE(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    if not args.no_wandb:
        wandb.watch(model)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    max_acc = 0.0
    set_seed(args.seed)

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args, label2vec = trainset.label2vec)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val', label2vec = valset.label2vec)

        if not args.no_wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc, \
                     'val/loss': val_loss, 'val/acc': val_acc, \
                     }, step=epoch)
                

        if val_acc > max_acc:
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc = val_acc
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})