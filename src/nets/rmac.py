import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResnetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResnetFeatureExtractor, self).__init__()
        pretrained_model = models.resnet50(pretrained=True)
        modules = list(pretrained_model.children())

        self.resnet_stem = nn.Sequential(*modules[:4])
        self.feat1, self.feat2, self.feat3, self.feat4 = modules[4:-2]

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Extracts feature maps from Resnet50 bottleneck layers.

        Args:
            x: B x 3 x H x W input tensor where B is the batch size.

        Returns:
            feat1: B x 256 x 56 x 56
            feat2: B x 512 x 28 x 28
            feat3: B x 1024 x 14 x 14
            feat4: B x 2048 x 7 x 7

        """
        stem_features = self.resnet_stem(x)
        feat1 = self.feat1(stem_features)  # B x 256 x 56 x 56
        feat2 = self.feat2(feat1)  # B x 512 x 28 x 28
        feat3 = self.feat3(feat2)  # B x 1024 x 14 x 14
        feat4 = self.feat4(feat3)  # B x 2048 x 7 x 7
        return feat1, feat2, feat3, feat4


class RmacFeatureExtractor(nn.Module):
    def __init__(self):
        super(RmacFeatureExtractor, self).__init__()
        self.resnet_extractor = ResnetFeatureExtractor()

    @staticmethod
    def get_rmac_features(feats, region_size):
        rmac_feats = F.max_pool2d(feats, int(region_size * 1.5), region_size, 0)
        return rmac_feats

    def forward(self, x):
        """
        Extract R-MAC features from each ConvNet layer.

        Args:
            x: B x 3 x H x W input tensor where B is the batch size.

        Returns:
            rmac_feats: B x 3840 x 3 x 3 R-MAC features
        """
        feat1, feat2, feat3, feat4 = self.resnet_extractor(x)
        rmac1 = self.get_rmac_features(feat1, 16)
        rmac2 = self.get_rmac_features(feat2, 8)
        rmac3 = self.get_rmac_features(feat3, 4)
        rmac4 = self.get_rmac_features(feat4, 2)
        rmac_feats = torch.cat([rmac1, rmac2, rmac3, rmac4], dim=1)  # 3840 x 3 x 3
        return rmac_feats
