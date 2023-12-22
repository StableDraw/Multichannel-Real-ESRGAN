import os
import torch
import wget
from collections import OrderedDict
from torch import nn as nn
from ...torchvision.models import vgg as vgg
from ..utils.registry import ARCH_REGISTRY

VGG_PRETRAIN_PATH = 'experiments\\pretrained_models\\'
VGG_MODEL_URL = "https://download.pytorch.org/models/"

VGG_MODEL_NAMES = {
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg13': 'vgg13-c768596a.pth',
    'vgg16': 'vgg16-397923af.pth',
    'vgg19': 'vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'vgg11_bn-6002323d.pth',
    'vgg13_bn': 'vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'vgg19_bn-c79401a0.pth'
}

NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn

def rebuild_vgg(in_channels, model_root, vgg_name):
    if not os.path.exists(model_root + vgg_name):
        wget.download(VGG_MODEL_URL + vgg_name, out = model_root)
    model = torch.load(model_root + vgg_name, map_location = torch.device('cpu'))
    first_layer_name = next(iter(model))
    first_layer = model[first_layer_name]
    first_layer_shape = list(first_layer.shape)
    first_layer_shape[1] = in_channels
    new_layer = torch.empty(first_layer_shape)
    for i in range(first_layer_shape[0]):
        s = 0
        cell = first_layer[i]
        for j in range(3):
            s += cell[j]
        s /= 3
        if in_channels == 1:
            new_layer[i][0] = s
        elif in_channels == 4:
            for j in range(3):
                new_layer[i][j] = cell[j]
            new_layer[i][3] = s
        else:
            for j in range(in_channels):
                new_layer[i][j] = (s + cell[j % 3]) / 2
    model[first_layer_name] = new_layer
    torch.save(model, model_root + str(in_channels) + "ch_" + vgg_name)
    

@ARCH_REGISTRY.register()
class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2,
                 in_channels=3):
        super(VGGFeatureExtractor, self).__init__()
        global VGG_PRETRAIN_PATH
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        DEFAULT_VGG_NAME = VGG_MODEL_NAMES[vgg_type]

        if in_channels == 3:
            VGG_PRETRAIN_PATH += DEFAULT_VGG_NAME
        else:
            VGG_PRETRAIN_ROOT_PATH = VGG_PRETRAIN_PATH
            VGG_PRETRAIN_PATH += str(in_channels) + "ch_" + DEFAULT_VGG_NAME
            if not os.path.exists(VGG_PRETRAIN_PATH):
                rebuild_vgg(in_channels, VGG_PRETRAIN_ROOT_PATH, DEFAULT_VGG_NAME)

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(weights = None, in_channels = in_channels)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location = lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(weights = vgg_type.upper() + "_Weights.IMAGENET1K_V1", in_channels = in_channels)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            original_mean_tensor = [0.485, 0.456, 0.406]
            original_std_tensor = [0.229, 0.224, 0.225]
            awerage_mean_value = 0.449
            awerage_std_value = 0.226
            if in_channels == 1:
                mean_tensor = [awerage_mean_value]
                std_tensor = [awerage_std_value]
            elif in_channels == 3:
                mean_tensor = original_mean_tensor
                std_tensor = original_std_tensor
            elif in_channels == 4:
                mean_tensor = original_mean_tensor + [awerage_mean_value]
                std_tensor = original_std_tensor + [awerage_std_value]
            else:
                mean_tensor = []
                std_tensor = []
                for i in range(in_channels):
                    mean_tensor += [(original_mean_tensor[i % 3] + awerage_mean_value) / 2]
                    std_tensor += [(original_std_tensor[i % 3] + awerage_std_value) / 2]
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor(mean_tensor).view(1, in_channels, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor(std_tensor).view(1, in_channels, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output
