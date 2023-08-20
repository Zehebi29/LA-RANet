import argparse
import torch
from torchvision import transforms
import os


def get_args():
    '''

    :return: args for the traning log
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--py_name", default='train_relation_attention.py', type=str, help=" ")
    parser.add_argument("--input_size", default=224, type=int, help=" ")
    parser.add_argument("--data_root", default='/data_path', type=str, help=" ")
    parser.add_argument("--batch_size", default=32, type=int, help=" ")
    parser.add_argument("--num_classes", default=5, type=int, help=" ")
    parser.add_argument("--model_structure", default='structure_name', type=str, help="Model name")
    parser.add_argument("--change_info", default='Add Relation attention module to Net', type=str, help="what has been changed change")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--log_path", default='./logs/structure_name', type=str, help="Path for saving logs")
    parser.add_argument("--pth_path", default='./runs/structure_name', type=str, help="Path for saving pth")
    parser.add_argument("--ssl_pth", default='./your_model.pth', type=str, help="Path for saving ssl pre-trained model")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=0, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--ssl_option", default=False, type=bool, help=" ")
    parser.add_argument("--ngpu", default=0, type=int, help="the nth gpu used for trainig")
    parser.add_argument("--resume_option", default=False, type=bool, help=" ")
    parser.add_argument("--resume_pth", default=' ', type=str, help="Path for saving ssl pre-trained model")


    args = parser.parse_args()
    return args


def test_kwargs():
    '''

    :return: **kwargs
    '''
    kwargs = {}
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    args = get_args()
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(args.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    mean, std = [0.5], [0.5]
    gray_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    kwargs['transforms'] = val_transform
    kwargs['gray_transforms'] = gray_transforms
    kwargs['num_class'] = args.num_classes
    kwargs['batch_size'] = args.batch_size
    kwargs['data_path'] = '/data_path'
    return kwargs


def ssl_args():
    '''

    :return: opt for model config and training
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet', help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default='../data_path/rectangle_mask', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', default='True', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--nBottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
    parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
    parser.add_argument('--nef', type=int, default=64, help='of encoder filters in first conv layer')
    parser.add_argument('--wtl2', type=float, default=0.998, help='0 means do not use else use with this weight')
    parser.add_argument('--wtlD', type=float, default=0.001, help='0 means do not use else use with this weight')
    parser.add_argument('--change_info', type=str, help='transforms changed')
    parser.add_argument('--train_file', type=str, help='train file name changed')

    opt = parser.parse_args()
    return opt