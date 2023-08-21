import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16_bn
from itertools import combinations


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=2):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        return self.sigmoid(avgout)


class RelationalAttention(nn.Module):
    def __init__(self, planes=64, n=36):
        """ Constructor
        Args:
            planes: output channel dimensionality.
            n: the number used for channels, default:9*8/2=36
        """
        super(RelationalAttention, self).__init__()
        self.a = [i for i in range(9)]
        self.combination = list(combinations(self.a, 2))
        self.group_conv = nn.Conv2d(2*n, n, kernel_size=1, bias=False, groups=n)
        self.bn_group = nn.BatchNorm2d(n)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(n, n, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n)

        self.channel_attention = ChannelAttentionModule(n)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        x = get_glcm_stack(avgout)

        for (i, (l1, l2)) in enumerate(self.combination):
            fea = torch.cat((x[:, l1, :, :].unsqueeze(dim=1), x[:, l2, :, :].unsqueeze(dim=1)), dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat((feas, fea), dim=1)
        feas = self.relu(self.bn_group(self.group_conv(feas)))

        feas = self.conv1(feas)
        feas = self.bn1(feas)

        feas = self.channel_attention(feas)*feas
        feas = torch.sum(feas, dim=1).unsqueeze(1)
        return self.sigmoid(feas)


def get_glcm_stack(img, step=1):
    gl3 = torch.cat((img[:, :, :, step:], img[:, :, :, -1 * step:]), dim=3)
    gl6 = torch.cat((img[:, :, step:, :], img[:, :, -1 * step:, :]), dim=2)
    gl7 = torch.cat((img[:, :, :step, :], img[:, :, :-1 * step, :]), dim=2)
    gl8 = torch.cat((img[:, :, :step, :], img[:, :, :-1 * step, :]), dim=2)
    trans_img = torch.cat((img, torch.cat((img[:, :, :, step:], img[:, :, :, -1 * step:]), dim=3), torch.cat((img[:, :, step:, :], img[:, :, -1 * step:, :]), dim=2),
                           torch.cat((gl3[:, :, step:, :], gl3[:, :, -1 * step:, :]), dim=2), torch.cat((img[:, :, :, :step], img[:, :, :, :-1 * step]), dim=3),
                           torch.cat((img[:, :, :step, :], img[:, :, :-1 * step, :]), dim=2), torch.cat((gl6[:, :, :, step:], gl6[:, :, :, -1 * step:]), dim=3),
                           torch.cat((gl7[:, :, :, step:], gl7[:, :, :, -1 * step:]), dim=3), torch.cat((gl8[:, :, :, :step], gl8[:, :, :, :-1 * step]), dim=3)), dim=1)
    return trans_img


class Net(nn.Module):
    def __init__(self, num_classes, pre_trained=False, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(Net, self).__init__()
        self.device = device
        self.backbone = vgg16_bn(pretrained=pre_trained)
        self.relationNet = RelationNetwork(cfg=[64, 128, 256, 512, 512], device=self.device)

        # build decoder
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(72128, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        fea0 = self.backbone.features[: 6](x)
        fea1 = self.backbone.features[: 13](x)
        fea2 = self.backbone.features[: 23](x)
        fea3 = self.backbone.features[: 33](x)
        fea4 = self.backbone.features[: 43](x)

        # reconstruct_pred = self.decoder(fea4)

        multi_scale_feature = [fea0, fea1, fea2, fea3, fea4]
        out = self.relationNet(multi_scale_feature)
        for i, xi in enumerate(out):
            if i == 0:
                avg_out = self.avgpool(xi)
            else:
                avg_out = torch.cat([avg_out, self.avgpool(xi)],dim=1)
        avg_out = torch.flatten(avg_out, 1)
        cls_pred = self.classifier(avg_out)
        return cls_pred


class RelationNetwork(nn.Module):
    def __init__(self, cfg, device):
        super(RelationNetwork, self).__init__()
        self.att_bulk = nn.ModuleList([RelationalAttention(cfg[i]).to(device) for i in range(len(cfg))])

    def forward(self, featrues):
        out = []
        for i, fea in enumerate(featrues):
            out.append(self.att_bulk[i](fea) * fea)
        return out
