# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import vovnet

from .mmdetection.vgg import VGG

@registry.BACKBONES.register("V-27-FPN")
@registry.BACKBONES.register("V-39-FPN")
@registry.BACKBONES.register("V-57-FPN")
@registry.BACKBONES.register("V-75-FPN")
@registry.BACKBONES.register("V-93-FPN")
def build_vovnet_fpn_backbone(cfg):
    body = vovnet.VoVNet(cfg)
    in_channels_stage = cfg.MODEL.VOVNET.OUT_CHANNELS
    out_channels = cfg.MODEL.VOVNET.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage,
            in_channels_stage * 2,
            in_channels_stage * 3,
            in_channels_stage * 4,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("V-27-FPN-RETINANET")
@registry.BACKBONES.register("V-39-FPN-RETINANET")
@registry.BACKBONES.register("V-57-FPN-RETINANET")
@registry.BACKBONES.register("V-75-FPN-RETINANET")
@registry.BACKBONES.register("V-93-FPN-RETINANET")
def build_vovnet_fpn__p3p7_backbone(cfg):
    body = vovnet.VoVNet(cfg)
    in_channels_stage = cfg.MODEL.VOVNET.OUT_CHANNELS
    out_channels = cfg.MODEL.VOVNET.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage * 4 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels    
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage * 2,
            in_channels_stage * 3,
            in_channels_stage * 4,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model
    
# VGG-16-FPN-RETINANET output:
# [B, 256, 84, 168]
# [B, 256, 42, 84]
# [B, 256, 21, 42]
# [B, 256, 11, 21]
# [B, 256, 6, 11]
@registry.BACKBONES.register("VGG-16-FPN-RETINANET")
def build_vgg_fpn_backbone(cfg):
    body = VGG(depth=16, with_last_pool=True, frozen_stages=2)
    in_channels_stage2 = 128    # default: cfg.MODEL.RESNETS.RES2_OUT_CHANNELS (256)
    out_channels = 256          # default: cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS (256)
    in_channels_p6p7 = in_channels_stage2 * 4 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 4, # in_channels_stage2 * 8
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model

@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
