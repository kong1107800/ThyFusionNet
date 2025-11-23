import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RobustContrastiveLoss(nn.Module):
    """
    鲁棒对比损失函数，处理4维特征图输入
    """

    def __init__(self, temperature=0.07, margin=0.5, k=5):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.margin = margin
        self.k = k

    @property
    def temperature(self):
        return torch.exp(self.log_temperature.clamp(min=math.log(1e-5), max=math.log(10.0)))

    def forward(self, ct_features, us_features, labels):
        """
        处理4维特征图输入 [batch_size, channels, height, width]
        """
        batch_size = ct_features.size(0)

        # 特征标准化和展平
        ct_flatten = F.normalize(ct_features.flatten(1), p=2, dim=1)  # [B, C*H*W]
        us_flatten = F.normalize(us_features.flatten(1), p=2, dim=1)  # [B, C*H*W]

        # 计算相似度矩阵
        sim_matrix = torch.mm(ct_flatten, us_flatten.t())  # [batch_size, batch_size]

        # 创建标签矩阵
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]

        # 正样本对：对角线（自身配对）和相同类别的非自身样本
        pos_mask = label_matrix.clone()
        pos_mask.fill_diagonal_(False)  # 排除自身配对

        # 负样本对：不同类别的样本
        neg_mask = ~label_matrix

        # 计算正样本损失
        pos_sim = sim_matrix[pos_mask]
        if len(pos_sim) > 0:
            pos_loss = -torch.log(torch.sigmoid((pos_sim - self.margin) / self.temperature)).mean()
        else:
            pos_loss = torch.tensor(0.0, device=ct_features.device)

        # 难负样本挖掘
        neg_sim = sim_matrix.clone()
        neg_sim[~neg_mask] = -float('inf')  # 将非负样本设为极小值

        # 选择最难的负样本（相似度最高的负样本）
        if self.k > 0:
            topk_values, _ = torch.topk(neg_sim, k=min(self.k, batch_size), dim=1)
            hard_neg_sim = topk_values[topk_values > -float('inf')]  # 过滤无效值

            if len(hard_neg_sim) > 0:
                neg_loss = torch.logsumexp(hard_neg_sim / self.temperature, dim=0) - math.log(self.k)
            else:
                neg_loss = torch.tensor(0.0, device=ct_features.device)
        else:
            neg_loss = torch.tensor(0.0, device=ct_features.device)

        # 总损失 = 正样本损失 + 负样本损失
        return pos_loss + neg_loss


class AdaptiveACE(nn.Module):
    """
    自适应对比-熵损失函数，处理4维特征图输入
    """

    def __init__(self, ce_weight_init=1.0, cont_weight_init=1.0, min_weight=0.1, ema_decay=0.9):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = RobustContrastiveLoss()

        # 可学习的权重参数
        self.log_ce_weight = nn.Parameter(torch.log(torch.tensor(ce_weight_init)))
        self.log_cont_weight = nn.Parameter(torch.log(torch.tensor(cont_weight_init)))

        # 权重约束参数
        self.min_weight = min_weight
        self.ema_decay = ema_decay

        # 注册缓冲区用于存储历史损失
        self.register_buffer('ce_ema', torch.tensor(0.0))
        self.register_buffer('cont_ema', torch.tensor(0.0))
        self.register_buffer('step', torch.tensor(0))

    @property
    def ce_weight(self):
        """使用softplus确保权重为正"""
        return F.softplus(self.log_ce_weight) + self.min_weight

    @property
    def cont_weight(self):
        """使用softplus确保权重为正"""
        return F.softplus(self.log_cont_weight) + self.min_weight

    def forward(self, outputs, labels, ct_features, us_features):
        """
        接受4维特征图输入 [batch_size, channels, height, width]
        """
        # 计算分类损失
        ce_loss = self.ce_loss(outputs, labels)

        # 计算对比损失 - 直接传入4维特征图
        cont_loss = self.contrastive_loss(ct_features, us_features, labels)

        # 更新EMA损失值
        if self.step == 0:
            self.ce_ema = ce_loss.detach()
            self.cont_ema = cont_loss.detach()
        else:
            self.ce_ema = self.ema_decay * self.ce_ema + (1 - self.ema_decay) * ce_loss.detach()
            self.cont_ema = self.ema_decay * self.cont_ema + (1 - self.ema_decay) * cont_loss.detach()

        self.step += 1

        # 计算加权损失
        weighted_ce = self.ce_weight * ce_loss
        weighted_cont = self.cont_weight * cont_loss

        # 总损失
        total_loss = weighted_ce + weighted_cont

        return total_loss

    def get_weights(self):
        """获取当前权重值（用于监控）"""
        return {
            'ce_weight': self.ce_weight.item(),
            'cont_weight': self.cont_weight.item(),
            'temperature': self.contrastive_loss.temperature.item()
        }