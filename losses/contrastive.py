import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import CircleLoss
from .pairwise import CirclePairLoss, CosFacePairLoss
from .triplet import TripletLoss


def build_metric_loss(metric_cfg):
    if metric_cfg.LOSS == 'CircleLoss':
        return CircleLoss(m=metric_cfg.LOSS_MARGIN, gamma=metric_cfg.LOSS_SCALE)
    elif metric_cfg.LOSS == 'TripletLoss':
        return TripletLoss(margin=metric_cfg.LOSS_MARGIN)
    elif metric_cfg.LOSS == 'PairCircleLoss':
        return CirclePairLoss(s=metric_cfg.LOSS_SCALE, m=metric_cfg.LOSS_MARGIN)
    elif metric_cfg.LOSS == 'PairCosFace':
        return CosFacePairLoss(s=metric_cfg.LOSS_SCALE, m=metric_cfg.LOSS_MARGIN)
    return None

def clip_loss(visual_embeds, lang_embeds, logit_scale=None, clip_loss_margin=0):
    sim_i_2_t = torch.matmul(visual_embeds, torch.t(lang_embeds))
    # batchxbatch, row1 is img1 vs lang1-n

    acc_sim = sim_i_2_t.clone().detach()
    sim_i_2_t = sim_i_2_t - (torch.eye(sim_i_2_t.size(0)).cuda() * clip_loss_margin)
    if logit_scale:
        sim_i_2_t = torch.mul(logit_scale, sim_i_2_t)
    sim_t_2_i = sim_i_2_t.t()
    # batchxbatch, row1 is lang1 vs img1-n

    loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda())
    # [similar, not, ..., not] vs 0
    # [not, similar, ..., not] vs 1
    # [not, not, ..., similar] vs n-1

    loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(sim_i_2_t.size(0)).cuda())
    loss = (loss_t_2_i + loss_i_2_t) / 2
    
    return loss, acc_sim

def view_loss(head_cfg, neg_sim, pos_sim=None, logit_scale_nl=None):
    if head_cfg.NLP_VIEW_LOSS == 'Triplet':
        delta = neg_sim - pos_sim + head_cfg.NLP_VIEW_LOSS_MARGIN
        if head_cfg.NLP_VIEW_SOFT:
            return F.softplus(logit_scale_nl * delta).mean()
        return F.relu(delta).mean()
    # contrastive
    if head_cfg.NLP_VIEW_SOFT:
        return F.softplus(logit_scale_nl * neg_sim).mean()
    return F.relu(neg_sim).mean()

