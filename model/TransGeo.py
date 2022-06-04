import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from .Deit import deit_small_distilled_patch16_224

class TransGeo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self,  args, base_encoder=None):
        """
        dim: feature dimension (default: 512)
        """
        super(TransGeo, self).__init__()
        self.dim = args.dim

        # create the encoders
        # num_classes is the output fc dimension

        if args.dataset == 'vigor':
            self.size_sat = [320, 320]
            self.size_sat_default = [320, 320]
            self.size_grd = [320, 640]
        elif args.dataset == 'cvusa':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]
        elif args.dataset == 'cvact':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]

        if args.sat_res != 0:
            self.size_sat = [args.sat_res, args.sat_res]
        if args.fov != 0:
            self.size_grd[1] = int(args.fov / 360. * self.size_grd[1])

        self.ratio = self.size_sat[0]/self.size_sat_default[0]
        base_model = deit_small_distilled_patch16_224

        self.query_net = base_model(crop=False, img_size=self.size_grd, num_classes=args.dim)
        self.reference_net = base_model(crop=args.crop, img_size=self.size_sat, num_classes=args.dim)
        self.polar = None

    def forward(self, im_q, im_k, delta=None, atten=None, indexes=None):
        if atten is not None:
            return self.query_net(im_q), self.reference_net(x=im_k, atten=atten)
        else:
            return self.query_net(im_q), self.reference_net(x=im_k, indexes=indexes)
