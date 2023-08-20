from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import cv2

from model.context_encoder_model import _netlocalD, _netG
import layer_aware_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../new_save/layer_segment/rectangle_mask', help='path to dataset')
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

opt = parser.parse_args()
print(opt)

try:
    os.makedirs("../new_save/result/train/cropped")
    os.makedirs("../new_save/result/train/real")
    os.makedirs("../new_save/result/train/recon")
    os.makedirs("../new_save/model")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                           )
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch = 0

netG = _netG(opt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
print(netG)

netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
print(netD)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

bz = int(opt.batchSize)
sz = int(opt.imageSize / 2)

# customized real_center cize


# real_center = torch.FloatTensor(bz, 3, sz, sz)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
    # real_center = real_center.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

# real_center = Variable(real_center)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

file_name_list = []
with open('./center_label_1.txt') as f:
    label_list = f.readlines()
    for i in range(len(label_list)):
        label_ = label_list[i].split(',')
        file_name_txt = label_[0].split('/')[-1]
        file_name_list.append(file_name_txt)
    f.close()

from torchvision import transforms
transform_center = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Scale(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])


for epoch in range(resume_epoch, opt.niter):
    loss_list = []
    loss_D = []
    loss_G = []
    error_ = 99999
    errorD_ = 99999
    errorG_ = 99999
    for i, data in enumerate(dataloader, 0):
        real_cpu, cls_label, _ = data
        input_cpu = input_cpu.cuda()
        input_cpu = Variable(input_cpu)
        # real_center_cpu = real_cpu[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2),
        #                   int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2)]
        for j, _path in enumerate(path):
            # real_cpu_slice = real_cpu[j, :, :, :].unsqueeze(0)

            path_name = _path.split('/')[-1]
            index = file_name_list.index(path_name)
            label_line = label_list[index]
            center_x = label_line.split(',')[1]
            center_y = label_line.split(',')[2].split('\n')[0]

            real_cpu_slice = cv2.imread(f'../EUS_together/data_full/{path_name}')
            real_cpu_slice = transform_center(real_cpu_slice).unsqueeze(0)
            real_center_cpu_slice = real_cpu_slice[:, :, int(center_x)-56:int(center_x)+56, int(center_y)-56:int(center_y)+56]

            if j == 0:
                real_center_cpu_new = real_center_cpu_slice
            else:
                real_center_cpu_new = torch.cat((real_center_cpu_new, real_center_cpu_slice), dim=0)

        real_center_cpu_new = real_center_cpu_new.cuda()
        real_center_cpu_new = Variable(real_center_cpu_new)

        batch_size = input_cpu.size(0)
        input_real.resize_(input_cpu.size()).copy_(input_cpu)
        input_cropped.resize_(input_cpu.size()).copy_(input_cpu)
        # real_center.resize_(input_cpu.size()).copy_(input_cpu)
        # input_cropped.data[:, 0,
        # int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
        # int(opt.imageSize / 4 + opt.overlapPred):int(
        #     opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
        # input_cropped.data[:, 1,
        # int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
        # int(opt.imageSize / 4 + opt.overlapPred):int(
        #     opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
        # input_cropped.data[:, 2,
        # int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
        # int(opt.imageSize / 4 + opt.overlapPred):int(
        #     opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 123.0 / 255.0 - 1.0

        # train with real
        netD.zero_grad()
        label.resize_(batch_size).fill_(real_label)

        output = netD(real_center_cpu_new)
        # label = output.unsqueeze(-1)
        print(epoch)
        label = label.unsqueeze(-1)
        assert output.shape == label.shape
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        fake = netG(input_cpu)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG_D = criterion(output, label)
        # errG_D.backward(retain_variables=True)

        # errG_l2 = criterionMSE(fake,real_center)
        wtl2Matrix = real_center_cpu_new.clone()
        wtl2Matrix.data.fill_(wtl2 * overlapL2Weight)
        wtl2Matrix.data[:, :, int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred),
        int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred)] = wtl2

        errG_l2 = (fake - real_center_cpu_new).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1 - wtl2) * errG_D + wtl2 * errG_l2
        loss_D.append(errG_D.cpu().detach())
        loss_G.append(errG_l2.cpu().detach())
        loss_list.append(errG.cpu().detach())

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG_D.item(), errG_l2.item(), D_x, D_G_z1,))
        if i % 100 == 0:
            vutils.save_image(real_center_cpu_new,
                              '../new_save/result/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,
                              '../new_save/result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = input_cropped.clone()
            recon_image.data[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2),
            int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2)] = fake.data
            vutils.save_image(recon_image.data,
                              '../new_save/result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))

    # do checkpointing
    import numpy as np
    error_final = np.mean(loss_list)
    errorD_final = np.mean(loss_D)
    errorG_final = np.mean(loss_G)

    if error_final < error_:
        error_ = error_final
        print(f'best error_overall: {error_}')
        torch.save({'epoch': epoch + 1,
                    'state_dict': netD.state_dict()},
                   f'../new_save/model/overall_error/netlocalD_error.pth')
        torch.save({'epoch': epoch + 1,
                    'state_dict': netG.state_dict()},
                   f'../new_save/model/overall_error/netG_error.pth')
    else:
        pass

    if errorD_final < errorD_:
        errorD_ = errorD_final
        print(f'best error_D: {errorD_}')

        torch.save({'epoch': epoch + 1,
                    'state_dict': netD.state_dict()},
                   f'../new_save/model/D_error/netlocalD_error.pth')
        torch.save({'epoch': epoch + 1,
                    'state_dict': netG.state_dict()},
                   f'../new_save/model/D_error/netG_error.pth')
    else:
        pass

    if errorG_final < errorG_:
        errorG_ = errorG_final
        print(f'best error_G: {errorG_}')
        torch.save({'epoch': epoch + 1,
                    'state_dict': netD.state_dict()},
                   f'../new_save/model/G_error/netlocalD_error.pth')
        torch.save({'epoch': epoch + 1,
                    'state_dict': netG.state_dict()},
                   f'../new_save/model/G_error/netG_error.pth')
    else:
        pass
