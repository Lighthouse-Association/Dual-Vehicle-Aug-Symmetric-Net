import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from ..losses import clip_loss, build_metric_loss, view_loss, cross_entropy
from ..benchmark import AverageMeter


class SiameseCollator(nn.Module):

    def __init__(self, cfg, encoder: nn.Module):
        super().__init__()
        print(f"====> SSM Siamese Collator:")
        self.encoder = encoder(cfg.MODEL)
        self.cfg = cfg
        self.metric_loss_f = build_metric_loss(cfg.MODEL.METRIC)
        self.acc_sim = None
        
    def get_acc_sim(self):
        return self.acc_sim

    def build_meters(self):
        print(f"===> Building meters")
        metric_learning = self.cfg.MODEL.METRIC.LOSS if self.cfg.MODEL.METRIC.LOSS else 'None'
        clip_losses = AverageMeter('Clip_Loss', ':.4e')
        cls_losses = AverageMeter('Cls_Loss', ':.4e')
        metric_losses = AverageMeter(metric_learning, ':.4e')
        nlp_view_losses = AverageMeter(f'Nlp_View_{self.cfg.MODEL.HEAD.NLP_VIEW_LOSS}_Loss', ':.4e')
        return {
            "clip_losses": clip_losses,
            "cls_losses": cls_losses,
            "metric_losses": metric_losses,
            "nlp_view_losses": nlp_view_losses
        }

    @torch.no_grad()
    def encode_text(self, *args, **kwargs):
        return self.encoder.encode_text(*args, **kwargs)
    
    @torch.no_grad()
    def encode_images(self, *args, **kwargs):
        return self.encoder.encode_images(*args, **kwargs)
    
    def forward(self, id_car, meters, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        pairs, logit_scale, cls_logits = outputs['pairs'], outputs['logit_scale'], outputs['cls_logits']

        logit_scale = logit_scale.mean().exp()

        con_clip_loss = torch.tensor(0., device='cuda')
        acc_sim = 0.
        if self.cfg.MODEL.NAME == 'dual-text-add' and not self.cfg.MODEL.HEAD.ADD_TRAIN or \
                self.cfg.MODEL.NAME == 'dual-text-cat' and not self.cfg.MODEL.HEAD.CAT_TRAIN:
            loss_pairs = pairs[:-1]  # not train, remove merge feat
        else:
            loss_pairs = pairs
            
        if self.cfg.MODEL.HEAD.CLIP_LOSS:
            for visual_embeds, lang_embeds in loss_pairs:
                t_clip_loss, acc_sim = clip_loss(visual_embeds, lang_embeds, logit_scale, \
                                                  self.cfg.MODEL.HEAD.CLIP_LOSS_MARGIN)
                con_clip_loss += t_clip_loss
        else:
            acc_sim = torch.matmul(pairs[-1][0], torch.t(pairs[-1][1])).detach()

        nlp_view_loss = torch.tensor(0., device='cuda')
        if 'view' in self.cfg.MODEL.NAME:
            view_lang_embeds, logit_scale_nl = outputs['view_nl']
            logit_scale_nl = logit_scale_nl.mean().exp()
            motion_vis_embeds, motion_lang_embeds = pairs[1]  # motion pair
            if self.cfg.MODEL.HEAD.NLP_VIEW_LOSS == 'Triplet':
                pos_sim = torch.diag(torch.matmul(motion_vis_embeds, motion_lang_embeds.T)).unsqueeze_(1)
                neg_sim = torch.diag(torch.matmul(motion_vis_embeds, view_lang_embeds.T)).unsqueeze_(1)
            else:
                pos_sim = None
                neg_sim = torch.diag(torch.matmul(motion_lang_embeds, view_lang_embeds.T)).unsqueeze_(1)
            nlp_view_loss = view_loss(self.cfg.MODEL.HEAD, neg_sim, pos_sim, logit_scale_nl)

        cls_loss = torch.tensor(0., device='cuda')
        for cls_logit in cls_logits:
            cls_loss += self.cfg.MODEL.HEAD.CLS_WEIGHT * \
                    cross_entropy(cls_logit, id_car.long().cuda(), epsilon=self.cfg.MODEL.HEAD.CE_EPSILON)
        # metric learning
        metric_loss = torch.tensor(0., device='cuda')
        if self.cfg.MODEL.HEAD.USE_METRIC_LOSS and self.metric_loss_f:
            for pair in loss_pairs:
                metric_loss += self.metric_loss_f(torch.cat(pair), torch.cat([id_car, id_car]).long().cuda())

        metric_loss *= self.cfg.MODEL.METRIC.METRIC_WEIGHT

        loss = con_clip_loss + cls_loss + metric_loss + nlp_view_loss

        batch_size = pairs[0][0].size(0)
        meters["clip_losses"].update(con_clip_loss.item(), batch_size)
        meters["cls_losses"].update(cls_loss.item(), batch_size)
        meters["metric_losses"].update(metric_loss.item(), batch_size)
        meters["nlp_view_losses"].update(nlp_view_loss.item(), batch_size)
        self.acc_sim = acc_sim
        return loss, acc_sim
        

class SiameseEnhancedCollator(nn.Module):

    def __init__(self, cfg, encoder: nn.Module):
        super().__init__()
        print(f"====> SSM Siamese Enhanced Collator:")
        self.encoder = encoder(cfg.MODEL, cfg.DATA)
        self.cfg = cfg
        self.metric_loss_f = build_metric_loss(cfg.MODEL.METRIC)
        self.acc_sim = None
        
    def get_acc_sim(self):
        return self.acc_sim

    def build_meters(self):
        # print(f"===> Building meters")
        # metric_learning = self.cfg.MODEL.METRIC.LOSS if self.cfg.MODEL.METRIC.LOSS else 'None'
        clip_losses = AverageMeter('Clip_Loss', ':.4e')
        # cls_losses = AverageMeter('Cls_Loss', ':.4e')
        
        type1 = AverageMeter('Type1_Loss', ':.4e')
        type2 = AverageMeter('Type2_Loss', ':.4e')
        motion1 = AverageMeter('Motion1_Loss', ':.4e')
        # motion2 = AverageMeter('Motion2_Loss', ':.4e')
        color1 = AverageMeter('Color1_Loss', ':.4e')
        color2 = AverageMeter('Color2_Loss', ':.4e')
        size1 = AverageMeter('Size1_Loss', ':.4e')
        # size2 = AverageMeter('Size2_Loss', ':.4e')
        inter = AverageMeter('Inter_Loss', ':.4e')
        return {
            "clip_losses": clip_losses,
            # "cls_losses": cls_losses,
            "type1": type1,
            "type2": type2,
            "motion1": motion1,
            # "motion2": motion2,
            "color1": color1,
            "color2": color2,
            "size1": size1,
            # "size2": size2,
            "inter": inter
        }

    @torch.no_grad()
    def encode_text(self, *args, **kwargs):
        return self.encoder.encode_text(*args, **kwargs)
    
    @torch.no_grad()
    def encode_images(self, *args, **kwargs):
        return self.encoder.encode_images(*args, **kwargs)
    
    def forward(self, id_car, meters, g_feats, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        pairs, logit_scale, cls_logits = outputs['pairs'], outputs['logit_scale'], outputs['cls_logits']
        vis_features = outputs['vis_features']
        proj_type1, proj_motion1, proj_color1, proj_size1, proj_car, proj_type2, \
            proj_color2, proj_mo, proj_inter = vis_features
            
        # Aggr losses
        type1, type2, motion1, color1, color2, size1, inter = g_feats
        
        type1_loss = F.cross_entropy(proj_type1, type1)
        type2_loss = F.cross_entropy(proj_type2, type2)
        motion1_loss = F.cross_entropy(proj_motion1, motion1)
        # motion2_loss = F.cross_entropy(proj_motion2, motion2)
        color1_loss = F.cross_entropy(proj_color1, color1)
        color2_loss = F.cross_entropy(proj_color2, color2)
        size1_loss = F.cross_entropy(proj_size1, size1)
        # size2_loss = F.cross_entropy(proj_size2, size2)
        inter_loss = F.binary_cross_entropy_with_logits(torch.squeeze(proj_inter), inter.float())
        
        aggr_loss = (type1_loss + type2_loss + motion1_loss + color1_loss + color2_loss \
                     + size1_loss + inter_loss) / self.cfg.MODEL.FEATS_AGGR_WEIGHT
        

        logit_scale = logit_scale.mean().exp()

        con_clip_loss = torch.tensor(0., device='cuda')
        acc_sim = 0.
        if self.cfg.MODEL.NAME == 'dual-text-add' and not self.cfg.MODEL.HEAD.ADD_TRAIN or \
                self.cfg.MODEL.NAME == 'dual-text-cat' and not self.cfg.MODEL.HEAD.CAT_TRAIN:
            loss_pairs = pairs[:-1]  # not train, remove merge feat
        else:
            loss_pairs = pairs
            
        if self.cfg.MODEL.HEAD.CLIP_LOSS:
            for visual_embeds, lang_embeds in loss_pairs:
                t_clip_loss, acc_sim = clip_loss(visual_embeds, lang_embeds, logit_scale, \
                                                  self.cfg.MODEL.HEAD.CLIP_LOSS_MARGIN)
                con_clip_loss += t_clip_loss
        else:
            acc_sim = torch.matmul(pairs[-1][0], torch.t(pairs[-1][1])).detach()

        nlp_view_loss = torch.tensor(0., device='cuda')
        if 'view' in self.cfg.MODEL.NAME:
            view_lang_embeds, logit_scale_nl = outputs['view_nl']
            logit_scale_nl = logit_scale_nl.mean().exp()
            motion_vis_embeds, motion_lang_embeds = pairs[1]  # motion pair
            if self.cfg.MODEL.HEAD.NLP_VIEW_LOSS == 'Triplet':
                pos_sim = torch.diag(torch.matmul(motion_vis_embeds, motion_lang_embeds.T)).unsqueeze_(1)
                neg_sim = torch.diag(torch.matmul(motion_vis_embeds, view_lang_embeds.T)).unsqueeze_(1)
            else:
                pos_sim = None
                neg_sim = torch.diag(torch.matmul(motion_lang_embeds, view_lang_embeds.T)).unsqueeze_(1)
            nlp_view_loss = view_loss(self.cfg.MODEL.HEAD, neg_sim, pos_sim, logit_scale_nl)

        cls_loss = torch.tensor(0., device='cuda')
        for cls_logit in cls_logits:
            cls_loss += self.cfg.MODEL.HEAD.CLS_WEIGHT * \
                    cross_entropy(cls_logit, id_car.long().cuda(), epsilon=self.cfg.MODEL.HEAD.CE_EPSILON)
        # metric learning
        metric_loss = torch.tensor(0., device='cuda')
        if self.cfg.MODEL.HEAD.USE_METRIC_LOSS and self.metric_loss_f:
            for pair in loss_pairs:
                metric_loss += self.metric_loss_f(torch.cat(pair), torch.cat([id_car, id_car]).long().cuda())

        metric_loss *= self.cfg.MODEL.METRIC.METRIC_WEIGHT

        loss = con_clip_loss + cls_loss + metric_loss + nlp_view_loss + aggr_loss

        batch_size = pairs[0][0].size(0)
        meters["clip_losses"].update(con_clip_loss.item(), batch_size)
        # meters["cls_losses"].update(cls_loss.item(), batch_size)
        meters["type1"].update(type1_loss.item(), batch_size)
        meters["type2"].update(type2_loss.item(), batch_size)
        meters["motion1"].update(motion1_loss.item(), batch_size)
        # meters["motion2"].update(motion2_loss.item(), batch_size)
        meters["color1"].update(color1_loss.item(), batch_size)
        meters["color2"].update(color2_loss.item(), batch_size)
        meters["size1"].update(size1_loss.item(), batch_size)
        # meters["size2"].update(size2_loss.item(), batch_size)
        meters["inter"].update(inter_loss.item(), batch_size)
        
        wandb.log({
            "clip_losses": meters["clip_losses"].avg,
            # "cls_losses": meters["cls_losses"].avg,
            "type1_loss": meters["type1"].avg,
            "type2_loss": meters["type2"].avg,
            "motion1_loss": meters["motion1"].avg,
            # "motion2_loss": meters["motion2"].avg,
            "color1_loss": meters["color1"].avg,
            "color2_loss": meters["color2"].avg,
            "size1_loss": meters["size1"].avg,
            # "size2_loss": meters["size2"].avg,
            "inter_loss": meters["inter"].avg
        })
        
        self.acc_sim = acc_sim
        return loss, acc_sim
    
