import time
from torch.nn import functional as torch_functional
from ..common.util_networks import Padder
from ..pipeline.transforms_pytorch import torch_onehot
from torchvision import models
from torch import nn
import numpy as np
import torch
import logging
import torchvision.models
import copy
log = logging.getLogger('exp')


class Correlation(nn.Module):

    @staticmethod
    def operation(a, b):
        """
        B x C x H x W
        """

        cor = Correlation.method1(a, b)
        # cor = Correlation.method2(a, b)
        # cor = torch.sum(a * b, dim=1, keepdim=True)

        return cor

    @staticmethod
    def method1(a, b):
        return torch.mean((a * b)**0.5, dim=1, keepdim=True)

    @staticmethod
    def method2(a, b):

        def corr2_coeff2(a, b):
            # Rowwise mean of input arrays & subtract from input arrays themeselves
            # a: BxC
            # b: BxC

            A_mA = a - a.mean(1).reshape(-1, 1)  # BxC
            B_mB = b - b.mean(1).reshape(-1, 1)  # BxC

            # Sum of squares across rows
            ssA = (A_mA**2).sum(1)              # Bx1
            ssB = (B_mB**2).sum(1)              # Bx1

            # Finally get corr coeff
            return (torch.mm(A_mA, B_mB.T) / torch.sqrt(torch.dot(ssA, ssB))).mean(1)

        B, C, H, W = a.shape
        cor = torch.zeros((B, 1, H, W)).cuda()

        for i in range(H):
            for j in range(W):
                cor[:, 0, i, j] = corr2_coeff2(
                    a[:, :, i, j], b[:, :, i, j]).ravel()

        return cor

    def forward(self, a, b):
        cor = Correlation.method1(a, b)
        # cor = Correlation.method2(a, b)
        # cor =  self.operation(self, a, b)
        return cor


class ResnetFeatures(nn.Module):
    # Resnet18

    LAYERS_RESNET18 = [3, 8, 15, 22, 29]

    def __init__(self, resnet_mode='resnet18', layers_to_extract=None, freeze=True, num_class=19, pretrained=True):
        super(ResnetFeatures, self).__init__()
        model_name = resnet_mode
        self.name = model_name
        self.num_class = num_class

        pretrained_model = torchvision.models.__dict__[
            model_name](pretrained)

        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.upsampe = nn.Upsample(scale_factor=4, mode='nearest')
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classify = nn.Linear(pretrained_model.fc.in_features, num_class)

        del pretrained_model

    @torch.no_grad()
    def forward(self, x):

        results = []
        x = self.conv1(x)
        # x = self.upsampe(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.upsampe(x)

        x = self.layer1(x)
        results.append(x)
        # print(x)

        x = self.layer2(x)
        results.append(x)
        # print(x)

        x = self.layer3(x)
        results.append(x)
        # print(x)

        x = self.layer4(x)
        results.append(x)
        # print(x)
        # print('#################################################################')

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classify(x)

        # return x
        return results


class CorrDifference01(nn.Module):

    class UpBlock(nn.Sequential):
        def __init__(self, in_channels, middle_channels, out_channels, b_upsample=True):

            modules = [
                nn.Conv2d(in_channels, middle_channels,
                          kernel_size=3, padding=1),
                nn.SELU(inplace=True),
                nn.Conv2d(middle_channels, middle_channels,
                          kernel_size=3, padding=1),
                nn.SELU(inplace=True),
            ]

            if b_upsample:
                modules += [
                    nn.ConvTranspose2d(
                        middle_channels, out_channels, kernel_size=2, stride=2),
                ]

            super().__init__(*modules)

    class CatMixCorr(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.conv_1x1 = nn.Conv2d(in_ch*2, in_ch, kernel_size=1)

        def forward(self, prev, feats_img, feats_rec):
            channels = [prev] if prev is not None else []

            channels += [
                Correlation.operation(feats_img, feats_rec),
                self.conv_1x1(torch.cat([feats_img, feats_rec], 1)),
            ]

            # print('cat', [ch.shape[1] for ch in channels])

            return torch.cat(channels, 1)

    def __init__(self, num_outputs=2, freeze=True):
        super().__init__()

        self.vgg_extractor = ResnetFeatures(
            resnet_mode='resnet18',
            layers_to_extract=ResnetFeatures.LAYERS_RESNET18[:4],
            freeze=freeze,
        )

        feat_channels = [512, 256, 128, 64]
        out_chans = [256, 256, 128, 64]
        prev_chans = [0] + out_chans[:-1]
        cmis = []
        decs = []

        for i, fc, oc, pc in zip(range(feat_channels.__len__(), 0, -1), feat_channels, out_chans, prev_chans):

            # print(i, fc)
            # print(i, fc+1+pc, oc, oc)

            cmi = self.CatMixCorr(fc)
            dec = self.UpBlock(fc+1+pc, oc, oc, b_upsample=(i != 1))

            cmis.append(cmi)
            decs.append(dec)

            # self.add_module('cmi_{i}'.format(i=i), cmi)
            # self.add_module('dec_{i}'.format(i=i), dec)

        self.cmis = nn.Sequential(*cmis)
        self.decs = nn.Sequential(*decs)
        self.final = nn.Conv2d(out_chans[-1], num_outputs, kernel_size=1)

    def forward(self, image, gen_image, **_):

        if gen_image.shape != image.shape:
            gen_image = gen_image[:, :, :image.shape[2], :image.shape[3]]

        if not self.training:
            padder = Padder(image.shape, 16)
            image, gen_image = (padder.pad(x) for x in (image, gen_image))

        with torch.no_grad():
            vgg_feats_img = self.vgg_extractor(image)
            vgg_feats_gen = self.vgg_extractor(gen_image)

        value = None
        num_steps = self.cmis.__len__()

        for i in range(num_steps):
            i_inv = num_steps-(i+1)
            value = self.decs[i](
                self.cmis[i](value, vgg_feats_img[i_inv], vgg_feats_gen[i_inv])
            )

        result = self.final(value)

        if not self.training:
            result = padder.unpad(result)

        return result


class ComparatorImageToLabels(nn.Module):

    class CatMix(nn.Module):
        def __init__(self, in_ch_img, in_ch_sem, out_ch):
            super().__init__()
            self.conv_1x1 = nn.Conv2d(
                in_ch_img + in_ch_sem, out_ch, kernel_size=1)

        def forward(self, prev, feats_img, feats_sem):
            channels = [prev] if prev is not None else []

            channels += [
                self.conv_1x1(torch.cat([feats_img, feats_sem], 1)),
            ]
            return torch.cat(channels, 1)

    class SemFeatures(nn.Sequential):

        def __init__(self, num_sem_classes, feat_channels_num_sem):
            self.num_sem_classes = num_sem_classes

            nonlinearily = nn.ReLU(True)

            layers = [nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(num_sem_classes,
                          feat_channels_num_sem[0], kernel_size=7, padding=0),
                nonlinearily,
            )]

            num_prev_ch = feat_channels_num_sem[0]
            for num_ch in feat_channels_num_sem[1:]:
                layers.append(nn.Sequential(
                    nn.Conv2d(num_prev_ch, num_ch, kernel_size=3,
                              stride=2, padding=1),
                    nonlinearily,

                ))
                num_prev_ch = num_ch

            super().__init__(*layers)

        def forward(self, labels):
            results = []
            value = torch_onehot(
                labels, self.num_sem_classes, dtype=torch.float32)
            # log.debug(f'discrepancy-onehot labels {labels.shape} {value.shape}')
            for slice in self:
                value = slice(value)
                results.append(value)

            return results

    def __init__(self, num_outputs=1, num_sem_classes=19, freeze=True):
        super().__init__()

        self.num_sem_classes = num_sem_classes

        self.vgg_extractor = ResnetFeatures(
            resnet_mode='resnet18',
            layers_to_extract=ResnetFeatures.LAYERS_RESNET18[:4],
            freeze=freeze,
        )

        feat_channels_num_vgg = [512, 256, 128, 64]
        feat_channels_num_sem = [256, 128, 64, 32]

        self.sem_extractor = self.SemFeatures(
            num_sem_classes, feat_channels_num_sem[::-1])

        out_chans = [256, 256, 128, 64]
        prev_chans = [0] + out_chans[:-1]

        cmis = []
        decs = []

        for i, fc, sc, oc, pc in zip(
            range(feat_channels_num_vgg.__len__(), 0, -1),
            feat_channels_num_vgg,
            feat_channels_num_sem,
            out_chans,
            prev_chans
        ):

            cmi = self.CatMix(fc, sc, fc)
            dec = CorrDifference01.UpBlock(fc+pc, oc, oc, b_upsample=(i != 1))

            cmis.append(cmi)
            decs.append(dec)

            # self.add_module('cmi_{i}'.format(i=i), cmi)
            # self.add_module('dec_{i}'.format(i=i), dec)

        self.cmis = nn.Sequential(*cmis)
        self.decs = nn.Sequential(*decs)
        self.final = nn.Conv2d(out_chans[-1], num_outputs, kernel_size=1)

    def forward(self, labels, image, **_):

        if not self.training:
            padder = Padder(image.shape, 16)
            image, labels = (padder.pad(x) for x in (image, labels))

        with torch.no_grad():
            vgg_feats_img = self.vgg_extractor(image)
            feats_sem = self.sem_extractor(labels)

        value = None
        num_steps = self.cmis.__len__()

        for i in range(num_steps):
            i_inv = num_steps-(i+1)
            value = self.decs[i](
                self.cmis[i](value, vgg_feats_img[i_inv], feats_sem[i_inv])
            )

        result = self.final(value)

        if not self.training:
            result = padder.unpad(result)

        return result


class ComparatorImageToGenAndLabels(nn.Module):

    class CatMixCorrWithSem(nn.Module):
        def __init__(self, in_ch_img, in_ch_sem, ch_out):
            super().__init__()
            self.conv_1x1 = nn.Conv2d(
                in_ch_img*2 + in_ch_sem, ch_out, kernel_size=1)

        def forward(self, prev, feats_img, feats_rec, feats_sem):
            channels = [prev] if prev is not None else []

            channels += [
                Correlation.operation(feats_img, feats_rec),
                self.conv_1x1(torch.cat([feats_img, feats_rec, feats_sem], 1)),
            ]

            return torch.cat(channels, 1)

    def __init__(self, num_outputs=2, num_sem_classes=19, freeze=True):
        super().__init__()

        self.num_sem_classes = num_sem_classes

        self.vgg_extractor = ResnetFeatures(
            resnet_mode='resnet18',
            layers_to_extract=ResnetFeatures.LAYERS_RESNET18[:4],
            freeze=freeze,
        )

        feat_channels_num_vgg = [512, 256, 128, 64]
        feat_channels_num_sem = [256, 128, 64, 32]

        self.sem_extractor = ComparatorImageToLabels.SemFeatures(
            num_sem_classes, feat_channels_num_sem[::-1])

        out_chans = [256, 256, 128, 64]
        prev_chans = [0] + out_chans[:-1]

        cmis = []
        decs = []

        for i, fc, sc, oc, pc in zip(
            range(feat_channels_num_vgg.__len__(), 0, -1),
            feat_channels_num_vgg,
            feat_channels_num_sem,
            out_chans,
            prev_chans
        ):

            cmi = self.CatMixCorrWithSem(fc, sc, fc)
            dec = CorrDifference01.UpBlock(
                fc + 1 + pc, oc, oc, b_upsample=(i != 1))

            cmis.append(cmi)
            decs.append(dec)

            # self.add_module('cmi_{i}'.format(i=i), cmi)
            # self.add_module('dec_{i}'.format(i=i), dec)

        self.cmis = nn.Sequential(*cmis)
        self.decs = nn.Sequential(*decs)
        self.final = nn.Conv2d(out_chans[-1], num_outputs, kernel_size=1)

    def forward(self, image, gen_image, labels, **_):

        if gen_image.shape != image.shape:
            gen_image = gen_image[:, :, :image.shape[2], :image.shape[3]]

        if not self.training:
            padder = Padder(image.shape, 16)
            image, gen_image, labels = (padder.pad(x.float()).type(
                x.dtype) for x in (image, gen_image, labels))

        # print(f'img {tuple(image.shape)} | gen {tuple(gen_image.shape)} | labels {tuple(labels.shape)}')
        # print(f'Label range {torch.min(labels)} ... {torch.max(labels)}')

        with torch.no_grad():
            vgg_feats_img = self.vgg_extractor(image)
            vgg_feats_gen = self.vgg_extractor(gen_image)

        feats_sem = self.sem_extractor(labels)

        value = None
        num_steps = self.cmis.__len__()

        for i in range(num_steps):
            i_inv = num_steps-(i+1)
            value = self.decs[i](
                self.cmis[i](value, vgg_feats_img[i_inv],
                             vgg_feats_gen[i_inv], feats_sem[i_inv])
            )
        result = self.final(value)

        if not self.training:
            result = padder.unpad(result)

        return result
