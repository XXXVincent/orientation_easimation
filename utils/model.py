import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#From https://github.com/tonylins/pytorch-mobilenet-v2
#model https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR

#MobileNet_v2
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def binary_cross_entropy_one_hot(output, target):
    """
    binary_cross_entropy in one hot form based on tf.keras
    the original implementation in tensorflow can be found via
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3526
    https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/ops/nn_impl.py#L109
    https://towardsdatascience.com/sigmoid-activation-and-binary-crossentropy-a-less-than-perfect-match-b801e130e31
    """
    eps = 1e-07
    output = torch.clamp(output, eps, 1 - eps)
    output = torch.log(output / (1. - output))

    return sigmoid_cross_entropy_with_logits(output, target)


def sigmoid_cross_entropy_with_logits(output, target):
    assert (output.shape == target.shape), \
        'output and target must have the same shape'

    zeros = torch.zeros_like(output, dtype=output.dtype)
    cond = (output >= zeros)
    relu_logits = torch.where(cond, output, zeros)
    neg_abs_logits = torch.where(cond, -output, output)
    sigmoid_logistic_loss = relu_logits - output * target \
                            + torch.log1p(torch.exp(neg_abs_logits))
    # return sigmoid_logistic_loss
    return sigmoid_logistic_loss.mean()


def OrientationLoss(y_pred, y_true):
    gt_square_sum = torch.pow(y_true, 2).sum(dim=2)
    valid_bin = gt_square_sum > 0.5
    sum_valid_bin = valid_bin.sum(dim=1)

    loss_per_angle = (y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
    loss_bin_sum = loss_per_angle.sum(dim=1)
    loss_per_bin = loss_bin_sum / sum_valid_bin.float()
    loss_per_batch = loss_per_bin.mean()
    loss = 2 - 2 * loss_per_batch

    return loss


#ori head
class Model(nn.Module):
    def __init__(self, features=None, in_channels=1280, bins=2, w=0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        self.avgpool = nn.AvgPool2d(2)
        self.ori_conv2d = nn.Conv2d(in_channels, 2*self.bins, 1)
        self.con_conv2d = nn.Conv2d(in_channels, self.bins, 1)

    def forward(self, x):
        x = self.features(x)  # 512 x 7 x 7
        x = self.avgpool(x)
        ori = self.ori_conv2d(x).view(-1, self.bins, 2)
        ori = F.normalize(ori, dim=2)
        conf = self.con_conv2d(x).view(-1, self.bins)
        conf = torch.softmax(conf, dim=1)

        return ori, conf
