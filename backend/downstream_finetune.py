import pandas as pd
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torchvision
import os
from model.RANet_basic import Net
from torch.autograd import Variable
from torchvision import transforms
import argparse
from tensorboardX import SummaryWriter
from args import get_args
import sys


def accuracy(input:Tensor, targs:Tensor):
    '''

    :param input: pred
    :param targs: ground truth
    :return:
    '''
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    tmp = (input == targs).float()
    acc = tmp.mean().cpu().detach().numpy()
    return acc

def load_weights(resume_pth, model):
    '''

    :param ssl_pth: the path of ssl pre-trained model
    :param model: RealationNet model
    :return:
    '''
    state_dict = torch.load(resume_pth)
    print('=================loading weights=================')
    model.load_state_dict(state_dict, strict=False)
    return model



def load_ssl_weights(ssl_pth, model):
    '''

    :param ssl_pth: the path of ssl pre-trained model
    :param model: RealationNet model
    :return:
    '''
    state_dict = torch.load(ssl_pth)
    st_dic = state_dict['state_dict']
    # print(model.features[0].weight)
    # print(model.features[1].weight)
    print('=================loading weights=================')
    model.load_state_dict(st_dic, strict=False)
    # print(model.features[0].weight)
    # print(model.features[1].weight)
    return model


def freeze_layer(model):
    '''

    :param model: RelationNet model
    :return:
    '''

    for name, param in model.named_parameters():
        if 'relationNet' in name or 'classifier' in name:
            pass
        else:
            # print(name, 'is freezed')
            print(f'\r Freezing layers {name}')
            param.requires_grad = False


def main():
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomCrop(args.input_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(degrees=60),
                                           transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset_train = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'train'), train_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(args.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    val_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'val'), val_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    model = Net(num_classes=args.num_classes).cuda()
    if args.resume_option:
        load_weights(args.resume_pth, model)
    elif args.ssl_option:
        load_ssl_weights(args.ssl_pth, model)
        # freeze_layer(model)
    else:
        pass
    print(model)
    print('=================end loading=================')

    os.makedirs(args.pth_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    writer = SummaryWriter(args.log_path)
    model.to(device)
    lr = .0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    minLoss, maxValacc = 99999, -99999
    new = -1
    # lr0 = 1e-5

    for epoch in range(args.epochs):
        print('EPOCH: ', epoch + 1, '/%s' % args.epochs)
        if epoch % 20 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        train_loss, train_acc = train(model, train_loader, optimizer, writer, epoch)
        val_loss, val_acc = val(model, val_loader, writer, epoch)

        print('Training loss:.......', train_loss)
        print('Validation loss:.....', val_loss)
        print('Training accuracy:...', train_acc)
        print('Validation accuracy..', val_acc)

        val_acc_ = val_acc

        if val_loss < minLoss:
            torch.save(model.state_dict(), args.pth_path + '/best_loss.pth')
            print(f'NEW BEST Val Loss: {val_loss} ........old best:{minLoss}')
            minLoss = val_loss
            print('')
        if val_acc_ > maxValacc:
            if new == -1:
                pass
            else:
                os.remove(args.pth_path + '/best_acc_%s.pth' % new)
            new = epoch
            torch.save(model.state_dict(), args.pth_path + '/best_acc_%s.pth' % new)
            print(f'NEW BEST Val Acc: {val_acc_} ........old best:{maxValacc}')
            maxValacc = val_acc_


def train(model, train_loader, optimizer, writer, epoch):
    train_acc = []
    running_loss = 0.0
    model.train()
    num = len(train_loader)
    count = 0
    for j, (images, labels, p) in enumerate(train_loader):
        # writer.add_image('train_image', images[0, :, :, :], global_step=epoch*num + j)
        images, labels = Variable(images.to(device)), Variable(labels.to(device))
        output = model(images)
        # print(labels)
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(output, labels)
        acc_item = accuracy(output, labels)
        train_acc.append(acc_item)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        count += 1
        writer.add_scalar('step_train_loss', loss.item(), epoch * num + j)
        writer.add_scalar('step_train_acc', acc_item, epoch * num + j)
    mean_train_loss = running_loss/count
    writer.add_scalar('train_loss', mean_train_loss, epoch)
    writer.add_scalar('train_acc', np.mean(train_acc), epoch)
    return mean_train_loss, np.mean(train_acc)


def val(model, val_loader, writer, epoch):
    val_acc = []
    model.eval()
    count = 0
    val_running_loss = 0.0
    total, correct = 0, 0
    for images, labels, path in val_loader:
        with torch.no_grad():
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            output = model(images)
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc.append(accuracy(output, labels))
            val_running_loss += loss.item()
            count += 1
    val_acc = correct/total
    mean_val_loss = val_running_loss / count
    img = images.detach()
    # writer.add_image('val_image', img[0, :, :, :], global_step=epoch)
    writer.add_scalar('val_loss', mean_val_loss, epoch)
    writer.add_scalar('val_acc', val_acc, epoch)
    return mean_val_loss, val_acc


def cfg_log(log_path):
    with open(log_path, 'r') as f:
        data = f.readlines()
    with open(log_path, 'a') as f:
        log_info = f"filename:{args.py_name}, " \
                   f"structure:{args.model_structure}, " \
                   f"chage_info:{args.change_info}," \
                   f"hyperprameters:{args.num_classes, args.batch_size, args.epochs, args.input_size, args.data_root}," \
                   f"log_path={args.log_path}, pth_path={args.pth_path}," \
                   f"self_pretrained option={args.ssl_option} \n"
        if log_info not in data:
            f.write(log_info)
        else:
            pass


if __name__ == '__main__':
    args = get_args()
    args.py_name = 'downstream_finetune.py'
    args.log_path = './logs/super_LA_RANet'
    args.pth_path = './runs/super_LA_RANet'
    args.model_structure = 'super_LA_RANet'
    args.resume_option = False
    args.resume_pth = './runs/SSL_relationNet_multihead_18/best_acc_49.pth'

    args.ssl_pth = '/home/zhb/Desktop/experiment/new_save/model_5/overall_error/netG.pth'

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    cfg_log('trainning-cfg-log.txt')
    main()