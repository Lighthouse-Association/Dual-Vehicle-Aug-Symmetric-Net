import torch
import torch.nn as nn

from ..losses import clip_loss, build_metric_loss, view_loss, cross_entropy
from ..benchmark import AverageMeter


class SiameseCollator(nn.Module):

    def __init__(self, cfg, encoder: nn.Module):
        super().__init__()
        print(f"====> SSM Siamese Collator:")
        self.encoder = encoder(cfg.MODEL)
        self.cfg = cfg
        self.metric_loss_f = build_metric_loss(cfg.MODEL.METRIC)

    def build_meters(self):
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
        return self.encoder.encode_image(*args, **kwargs)
    
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
            return loss, acc_sim
        
