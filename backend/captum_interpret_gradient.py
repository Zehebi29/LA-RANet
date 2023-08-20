import os
from torch.utils.data import DataLoader
os.getcwd()
from torchvision import datasets, transforms
from args import get_args, test_kwargs
from model.RANet_basic import Net
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def interpret(model, device, _transforms, **kwargs):
    transforms = _transforms
    data_path = kwargs['data_path']
    bz = kwargs['batch_size']
    num_class = kwargs['num_class']
    labelnumber = [i for i in range(num_class)]
    test_valid = datasets.ImageFolder(root=data_path,
                                      transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_valid, batch_size=bz,
                                              shuffle=False,
                                              num_workers=0)

    model.eval()
    method = kwargs['method']
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            img_name = data[2][0].split('/')[-1]
            img_name = img_name.split('.')[0]
            outputs = model(inputs)
            model2 = nn.DataParallel(model.to(device))
            outputs = F.softmax(outputs, dim=1)
            prediction_score, pred_label_idx = torch.topk(outputs, 1)
            pred_label_idx.squeeze_()

            integrated_gradients = IntegratedGradients(model2)

            attributions_ig = integrated_gradients.attribute(inputs, target=pred_label_idx, n_steps=200)
            default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                             [(0, '#ffffff'),
                                                              (0.25, '#000000'),
                                                              (1, '#000000')], N=256)
            '''_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                         np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                         method='blended_heat_map',
                                         cmap='Reds',
                                         show_colorbar=True,
                                         sign='positive',
                                         outlier_perc=1)'''
            target_category = int(labels[0])

            kwargs = {}
            kwargs['method_name'] = method
            kwargs['fig_name'] = img_name

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'heat_map',
                'all',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
                )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'heat_map',
                'positive',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
                )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'heat_map',
                'absolute_value',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
                )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'blended_heat_map',
                'all',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
                )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'blended_heat_map',
                'positive',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
                )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'blended_heat_map',
                'absolute_value',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
                )


            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'masked_image',
                'positive',
                cmap='Greens',
                show_colorbar=True,
                **kwargs
                )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'alpha_scaling',
                'positive',
                cmap='Blues',
                show_colorbar=True,
                **kwargs
                )
            ''' 
            heat_map = 1
            blended_heat_map = 2
            original_image = 3
            masked_image = 4
            alpha_scaling = 5
            '''

            kwargs = {}
            kwargs['method_name'] = method
            kwargs['fig_name'] = img_name

            noise_tunnel = NoiseTunnel(integrated_gradients)

            attributions_ig_nt = noise_tunnel.attribute(inputs, nt_samples=20, nt_samples_batch_size=5, nt_type='smoothgrad_sq',
                                                          target=pred_label_idx)

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'heat_map',
                'all',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
            )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'heat_map',
                'positive',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
            )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'heat_map',
                'absolute_value',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
            )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'blended_heat_map',
                'all',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
            )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'blended_heat_map',
                'positive',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
            )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'blended_heat_map',
                'absolute_value',
                cmap='Reds',
                show_colorbar=True,
                **kwargs
            )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'masked_image',
                'positive',
                cmap='Greens',
                show_colorbar=True,
                **kwargs
            )

            _ = viz.visualize_image_attr(
                np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                'alpha_scaling',
                'positive',
                cmap='Blues',
                show_colorbar=True,
                **kwargs
            )


def load_model(model, state_dict, device):
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def model_choose(name):
    if name == 'model_name':
        net = Net(num_classes=5)
        params = torch.load('../path/your_model_name.pth')
        model = load_model(net, params, device)
    # elif name == 'model_name_2':
    #   initialization
    return model