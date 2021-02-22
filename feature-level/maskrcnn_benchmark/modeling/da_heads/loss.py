"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import consistency_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from ..utils import cat

class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.loss_type = cfg.MODEL.DA_HEADS.IMG_LOSS_TYPE
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.pooler = pooler
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)
        self.focal_loss = SigmoidFocalLoss(cfg.MODEL.RETINANET.LOSS_GAMMA, cfg.MODEL.RETINANET.LOSS_ALPHA)
        
    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image.get_field('is_source')
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        return masks

    def __call__(self, da_img, att_src, att_trg, da_img_cst_features, da_ica_cst_features, targets):
        """
        Arguments:
            da_img (list[Tensor])
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
        """
        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)
        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        da_img_losses = []
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
            # da_img_flattened.append(da_img_per_level)
            # da_img_labels_flattened.append(da_img_label_per_level)
            if self.loss_type == 0:
              da_img_loss = F.binary_cross_entropy_with_logits(
                  da_img_per_level, da_img_label_per_level
              )
            elif self.loss_type == 1:
              da_img_loss = self.focal_loss(
                  da_img_per_level, da_img_label_per_level
              )
            da_img_losses.append(da_img_loss) # multi level loss
        # da_img_flattened = torch.cat(da_img_flattened, dim=1)
        # da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=1)
        # da_img_flattened = torch.cat(da_img_flattened, dim=0)
        # da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
        # da_img_loss = F.binary_cross_entropy_with_logits(
        #     da_img_flattened, da_img_labels_flattened
        # )
        da_ica_labels_flattened = torch.cat((torch.ones(1, att_src.size(-1)), torch.zeros(1, att_trg.size(-1))), -1)
        da_att_features = torch.cat((att_src, att_trg), -1)
        ica_loss = F.binary_cross_entropy_with_logits(
            da_att_features.cuda(), da_ica_labels_flattened.type(torch.cuda.FloatTensor)
        )
        
        da_consist_loss = consistency_loss(da_img_cst_features, da_att_cst_features.cuda(), da_ica_labels_flattened.squeeze().type(torch.cuda.FloatTensor))

        return da_img_losses, ica_loss, da_consist_loss

def make_da_heads_loss_evaluator(cfg):
    loss_evaluator = DALossComputation(cfg)
    return loss_evaluator
