import torch
import torch.nn as nn
import torch.nn.functional as F
from Re_MaxViT_skip_HAPE_PE_LSF_AGFusion_ADAACE import Re_MaxViT, tiny_args

###############################################################################
# 1. Gated Channel Fusion
###############################################################################
class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, ct, us):
        x = torch.cat([ct, us], dim=1)     # [B,2C,H,W]
        g = self.gate(x)                   # [B, C,H,W]
        return g * ct + (1 - g) * us       # [B, C,H,W]

class ThyroidFusionGated(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ct_resnet = Re_MaxViT(tiny_args)
        self.us_resnet = Re_MaxViT(tiny_args)
        self.fuser     = GatedFusion(in_channels=512)
        self.pool      = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(512, num_classes)

    def forward(self, ct_img, us_img):
        ct_x = F.gelu(self.ct_resnet(ct_img))
        us_x = F.gelu(self.us_resnet(us_img))
        fused = self.fuser(ct_x, us_x)
        out   = self.fc(self.pool(fused).flatten(1))
        return out, ct_x, us_x

###############################################################################
# 2. Cross-Modal Attention Fusion
###############################################################################
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, feat_channels, num_heads=4):
        super().__init__()
        C = feat_channels; H = num_heads
        self.heads = H
        self.scale = (C//H)**-0.5
        self.qkv_ct = nn.Conv2d(C, C*3, 1, bias=False)
        self.qkv_us = nn.Conv2d(C, C*3, 1, bias=False)
        self.proj   = nn.Conv2d(2*C, C, 1)

    def forward(self, ct, us):
        B,C,H,W = ct.shape
        q1,k1,v1 = self.qkv_ct(ct).chunk(3,1)
        q2,k2,v2 = self.qkv_us(us).chunk(3,1)

        def reshape(x):
            B,C,H,W = x.shape
            x = x.view(B, self.heads, C//self.heads, H*W)
            return x.permute(0,1,3,2)  # [B,heads,HW,Ch]

        # CT queries US
        q1,k_u,v_u = reshape(q1), reshape(k2), reshape(v2)
        attn1      = torch.softmax((q1 @ k_u.transpose(-1,-2))*self.scale, dim=-1)
        out1       = (attn1 @ v_u).permute(0,1,3,2).contiguous().view(B,C,H,W)
        # US queries CT
        q2,k_c,v_c = reshape(q2), reshape(k1), reshape(v1)
        attn2      = torch.softmax((q2 @ k_c.transpose(-1,-2))*self.scale, dim=-1)
        out2       = (attn2 @ v_c).permute(0,1,3,2).contiguous().view(B,C,H,W)

        fused = torch.cat([ct+out1, us+out2], dim=1)
        return self.proj(fused)

class ThyroidFusionCrossAttn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ct_resnet = Re_MaxViT(tiny_args)
        self.us_resnet = Re_MaxViT(tiny_args)
        self.fuser     = CrossModalAttentionFusion(512)
        self.pool      = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(512, num_classes)

    def forward(self, ct_img, us_img):
        ct_x = F.gelu(self.ct_resnet(ct_img))
        us_x = F.gelu(self.us_resnet(us_img))
        fused = self.fuser(ct_x, us_x)
        out   = self.fc(self.pool(fused).flatten(1))
        return out, ct_x, us_x

###############################################################################
# 3. Bi-Modal Squeeze-and-Excitation Fusion
###############################################################################
class BiModalSEFusion(nn.Module):
    def __init__(self, feat_channels, reduction=16):
        super().__init__()
        C2 = feat_channels * 2
        self.fc = nn.Sequential(
            nn.Linear(C2, C2//reduction, bias=False),
            nn.GELU(),
            nn.Linear(C2//reduction, C2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, ct, us):
        B,C,H,W = ct.shape
        x = torch.cat([ct, us], dim=1)    # [B,2C,H,W]
        s = x.mean(dim=[2,3])             # [B,2C]
        w = self.fc(s).view(B,2*C,1,1)    # [B,2C,1,1]
        x = x * w                         # [B,2C,H,W]
        return x[:,:C], x[:,C:]           # split

class ThyroidFusionSE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ct_resnet = Re_MaxViT(tiny_args)
        self.us_resnet = Re_MaxViT(tiny_args)
        self.fuser     = BiModalSEFusion(512)
        self.pool      = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(512, num_classes)

    def forward(self, ct_img, us_img):
        ct_x = F.gelu(self.ct_resnet(ct_img))
        us_x = F.gelu(self.us_resnet(us_img))
        ct_f, us_f = self.fuser(ct_x, us_x)
        fused = 0.5*(ct_f+us_f)
        out   = self.fc(self.pool(fused).flatten(1))
        return out, ct_x, us_x

###############################################################################
# 4. Bilinear Pooling Fusion
###############################################################################
class BilinearFusion(nn.Module):
    def __init__(self, feat_channels, rank=64):
        super().__init__()
        self.proj_ct = nn.Linear(feat_channels, rank, bias=False)
        self.proj_us = nn.Linear(feat_channels, rank, bias=False)
        self.fc      = nn.Linear(rank, feat_channels)

    def forward(self, ct, us):
        B,C,H,W = ct.shape
        ct_v = ct.mean(dim=[2,3])  # [B,C]
        us_v = us.mean(dim=[2,3])  # [B,C]
        x1   = self.proj_ct(ct_v)
        x2   = self.proj_us(us_v)
        z    = F.gelu(x1 * x2)     # [B,rank]
        w    = torch.sigmoid(self.fc(z)).view(B,C,1,1)
        return ct*w + us*(1-w)

class ThyroidFusionBilinear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ct_resnet = Re_MaxViT(tiny_args)
        self.us_resnet = Re_MaxViT(tiny_args)
        self.fuser     = BilinearFusion(512)
        self.pool      = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(512, num_classes)

    def forward(self, ct_img, us_img):
        ct_x  = F.gelu(self.ct_resnet(ct_img))
        us_x  = F.gelu(self.us_resnet(us_img))
        fused = self.fuser(ct_x, us_x)
        out   = self.fc(self.pool(fused).flatten(1))
        return out, ct_x, us_x

###############################################################################
# 5. Multi-Scale Feature Fusion
#    (Assumes backbone returns two scales)
###############################################################################
class MultiScaleFusion(nn.Module):
    def __init__(self, feat_channels=512):
        super().__init__()
        self.red1_ct = nn.Conv2d(256, feat_channels, 1)
        self.red2_ct = nn.Conv2d(512, feat_channels, 1)
        self.red1_us = nn.Conv2d(256, feat_channels, 1)
        self.red2_us = nn.Conv2d(512, feat_channels, 1)
        self.gate    = nn.Conv2d(feat_channels*2, feat_channels, 1)

    def forward(self, ct_feats, us_feats):
        ct1, ct2 = ct_feats
        us1, us2 = us_feats
        r1_ct = F.gelu(self.red1_ct(ct1))
        r2_ct = F.gelu(self.red2_ct(ct2))
        r1_us = F.gelu(self.red1_us(us1))
        r2_us = F.gelu(self.red2_us(us2))
        # upsample
        r2_ct = F.interpolate(r2_ct, size=r1_ct.shape[2:], mode='bilinear', align_corners=False)
        r2_us = F.interpolate(r2_us, size=r1_ct.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([r1_ct+r2_ct, r1_us+r2_us], dim=1)
        return F.gelu(self.gate(x))

class ThyroidFusionMultiScale(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ct_resnet = Re_MaxViT(tiny_args)  # must return (feat1,feat2)
        self.us_resnet = Re_MaxViT(tiny_args)
        self.fuser     = MultiScaleFusion(512)
        self.pool      = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(512, num_classes)

    def forward(self, ct_img, us_img):
        ct_f1, ct_f2 = self.ct_resnet(ct_img)
        us_f1, us_f2 = self.us_resnet(us_img)
        fused = self.fuser((ct_f1, ct_f2), (us_f1, us_f2))
        out   = self.fc(self.pool(fused).flatten(1))
        return out, ct_f2, us_f2

###############################################################################
# 6. Attention-Gated Fusion
###############################################################################
class AttentionGatedFusion(nn.Module):
    def __init__(self, feat_channels):
        super().__init__()
        self.theta = nn.Conv2d(feat_channels, feat_channels//2, 1, bias=False)
        self.phi   = nn.Conv2d(feat_channels, feat_channels//2, 1, bias=False)
        self.psi   = nn.Conv2d(feat_channels//2, 1, 1, bias=False)

    def forward(self, ct, us):
        x     = ct + us
        t     = F.gelu(self.theta(x))
        p     = F.gelu(self.phi(x))
        attn  = torch.sigmoid(self.psi(t + p))  # [B,1,H,W]
        return ct*attn + us*(1-attn)

class ThyroidFusionGate(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ct_resnet = Re_MaxViT(tiny_args)
        self.us_resnet = Re_MaxViT(tiny_args)
        self.fuser     = AttentionGatedFusion(512)
        self.pool      = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(512, num_classes)

    def forward(self, ct_img, us_img):
        ct_x  = F.gelu(self.ct_resnet(ct_img))
        us_x  = F.gelu(self.us_resnet(us_img))
        fused = self.fuser(ct_x, us_x)
        out   = self.fc(self.pool(fused).flatten(1))
        return out, ct_x, us_x

###############################################################################
# 7. Cross-Modal Contrastive Pretraining Fusion
###############################################################################
class ContrastivePretrainFusion(nn.Module):
    def __init__(self, feat_channels, emb_dim=256):
        super().__init__()
        self.project_ct = nn.Sequential(
            nn.Linear(feat_channels, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.project_us = nn.Sequential(
            nn.Linear(feat_channels, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, ct_feat, us_feat):
        ct_vec = ct_feat.mean(dim=[2,3])
        us_vec = us_feat.mean(dim=[2,3])
        z_ct   = F.normalize(self.project_ct(ct_vec), dim=-1)
        z_us   = F.normalize(self.project_us(us_vec), dim=-1)
        return z_ct, z_us

class ThyroidFusionContrastive(nn.Module):
    def __init__(self, num_classes, emb_dim=256):
        super().__init__()
        self.ct_resnet  = Re_MaxViT(tiny_args)
        self.us_resnet  = Re_MaxViT(tiny_args)
        self.con_prep   = ContrastivePretrainFusion(512, emb_dim)
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, ct_img, us_img):
        ct_x = F.gelu(self.ct_resnet(ct_img))
        us_x = F.gelu(self.us_resnet(us_img))
        z_ct, z_us = self.con_prep(ct_x, us_x)
        fused      = torch.cat([ct_x, us_x], dim=1)
        logits     = self.classifier(fused)
        return logits, z_ct, z_us
