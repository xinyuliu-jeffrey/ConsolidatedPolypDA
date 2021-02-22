# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer

from .loss import make_da_heads_loss_evaluator

class HAAHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(HAAHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.gn1 = nn.GroupNorm(32, 512)
        self.active = nn.LeakyReLU()
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.conv_attmap = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        for l in [self.conv1_da, self.conv2_da, self.conv_attmap]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = self.conv1_da(feature)
            t = self.gn1(t)
            t = self.active(t)
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out = torch.max(t, dim=1, keepdim=True)[0]
            scale = torch.cat([avg_out, max_out], dim=1)
            sp_x = self.conv_attmap(scale)
            t = t * torch.sigmoid(sp_x)
            t = F.relu(t)
            img_features.append(self.conv2_da(t))
        return img_features

class ICAHead(nn.Module):
    """
    Add ICA Head
    """
    def __init__(self, in_channels):
        super(ICAHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x, da_proposals):
        if self.training:
            src_proposals = da_proposals[0].bbox
            trg_proposals = da_proposals[1].bbox
            src_proposals_centers_x = (src_proposals[:, None, 0] + src_proposals[:, None, 2]) / 16
            src_proposals_centers_y = (src_proposals[:, None, 1] + src_proposals[:, None, 3]) / 16
            trg_proposals_centers_x = (trg_proposals[:, None, 0] + trg_proposals[:, None, 2]) / 16
            trg_proposals_centers_y = (trg_proposals[:, None, 1] + trg_proposals[:, None, 3]) / 16
            att_src = torch.zeros(1, len(src_proposals_centers_x))
            att_trg = torch.zeros(1, len(trg_proposals_centers_x))
            x = x[2]
            src_fea = x[0, ...].unsqueeze(0)
            src_fea = F.relu(self.conv1_da(src_fea))
            src_fea = self.conv2_da(src_fea)
            src_fea = src_fea.squeeze()
            h_s, w_s = src_fea.size()
            # clamp to avoid overrange
            src_proposals_centers_x = torch.clamp(src_proposals_centers_x, 0, w_s-1)
            src_proposals_centers_y = torch.clamp(src_proposals_centers_y, 0, h_s-1)
            trg_fea = x[1, ...].unsqueeze(0)
            trg_fea = F.relu(self.conv1_da(trg_fea))
            trg_fea = self.conv2_da(trg_fea)
            trg_fea = trg_fea.squeeze()
            h_t, w_t = trg_fea.size()
            # clamp to avoid overrange
            trg_proposals_centers_x = torch.clamp(trg_proposals_centers_x, 0, w_t-1)
            trg_proposals_centers_y = torch.clamp(trg_proposals_centers_y, 0, h_t-1)
            for i, src_ctr in enumerate(src_proposals_centers_x):
                loc = int(src_proposals_centers_y[i]), int(src_proposals_centers_x[i])
                att_src[0, i] = src_fea[loc]
            for i, trg_ctr in enumerate(trg_proposals_centers_x):
                loc = int(trg_proposals_centers_y[i]), int(trg_proposals_centers_x[i])
                att_trg[0, i] = trg_fea[loc]
            return att_src, att_trg
        else:
            return None, None

class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(DomainAdaptationModule, self).__init__()

        self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith('V') else res2_out_channels * stage2_relative_factor
        
        self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        
        self.img_weight = cfg.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT
        # self.haa_loss_weight = cfg.MODEL.DA_HEADS.HAA_LOSS_WEIGHT
        self.gcr_loss = cfg.MODEL.DA_HEADS.GCR_LOSS
        self.ica_loss = cfg.MODEL.DA_HEADS.ICA_LOSS
        self.p3 = cfg.MODEL.DA_HEADS.HAA_LOSS_P3
        self.p4 = cfg.MODEL.DA_HEADS.HAA_LOSS_P4
        self.p5 = cfg.MODEL.DA_HEADS.HAA_LOSS_P5
        self.p6 = cfg.MODEL.DA_HEADS.HAA_LOSS_P6
        self.p7 = cfg.MODEL.DA_HEADS.HAA_LOSS_P7
        self.ica_weight = cfg.MODEL.DA_HEADS.ICA_LOSS_WEIGHT
        self.gcr_weight = cfg.MODEL.DA_HEADS.GCR_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_att = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_img_cst = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_att_cst = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.haahead = HAAHead(in_channels)
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)
        self.icahead = ICAHead(in_channels)

    def forward(self, img_features, da_proposals, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if self.training:
            img_grl_fea = [self.grl_img(fea) for fea in img_features]
            da_img_features = self.haahead(img_grl_fea)
        
            att_grl_fea = [self.grl_att(fea) for fea in img_features]
            att_src, att_trg = self.icahead(att_grl_fea, da_proposals)
        
            img_grl_cst = [self.grl_img_cst(fea) for fea in img_features]
            da_img_cst_features = self.haahead(img_grl_cst)
            att_grl_fea = [self.grl_att_cst(fea) for fea in img_features]
            att_src, att_trg = self.icahead(att_grl_fea, da_proposals)
            da_ica_cst_features = torch.cat((att_src, att_trg), -1).permute(1,0)
            da_img_cst_features = [fea.sigmoid() for fea in da_img_cst_features]
            da_ica_cst_features = da_ica_cst_features.sigmoid()
        
        
            da_img_losses, ica_loss, gcr_loss = self.loss_evaluator(
                da_img_features, att_src, att_trg, da_img_cst_features, da_ica_cst_features, targets
            )
            losses = {}
            if self.img_weight > 0:
                losses["loss_haa_p3"] = self.p3 * da_img_losses[0]
                losses["loss_haa_p4"] = self.p4 * da_img_losses[1]
                losses["loss_haa_p5"] = self.p5 * da_img_losses[2]
                losses["loss_haa_p6"] = self.p6 * da_img_losses[3]
                losses["loss_haa_p7"] = self.p7 * da_img_losses[4]
                if self.ica_loss:
                  losses["ica_loss"] = self.ica_weight * ica_loss
                if self.gcr_loss:
                  losses["gcr_loss"] = self.gcr_weight * gcr_loss
            return losses
        return {}

def build_da_heads(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule(cfg)
    return []
