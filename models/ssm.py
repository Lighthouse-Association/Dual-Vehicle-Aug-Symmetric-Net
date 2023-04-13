import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel
from .senet import se_resnext50_32x4d
from .efficientnet import EfficientNet
from .resnest import resnest50d
from ..losses import build_softmax_cls

# baseline for test

supported_img_encoders = ["se_resnext50_32x4d","efficientnet-b2","efficientnet-b3"]


class SiameseLocalandMotionModelBIG_DualTextCat(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        embed_dim = self.model_cfg.EMBED_DIM
        double_embed_dim = 2 * embed_dim
        merge_dim = self.model_cfg.MERGE_DIM

        # visual model
        if self.model_cfg.IMG_ENCODER in supported_img_encoders:
            if self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.vis_backbone = se_resnext50_32x4d()
                self.vis_backbone_bk = se_resnext50_32x4d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Conv2d(self.img_in_dim, embed_dim, kernel_size=1)
                self.domian_vis_fc_bk = nn.Conv2d(self.img_in_dim, embed_dim, kernel_size=1)
            elif self.model_cfg.IMG_ENCODER == "resnest50":
                self.vis_backbone = resnest50d()
                self.vis_backbone_bk = resnest50d()
                self.img_in_dim = 2048
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
            else:
                self.vis_backbone = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.vis_backbone_bk = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.img_in_dim = self.vis_backbone.out_channels
                self.domian_vis_fc = nn.Linear(self.img_in_dim, embed_dim)
                self.domian_vis_fc_bk = nn.Linear(self.img_in_dim, embed_dim)
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        # text model
        self.bert_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.lang_car_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.lang_mo_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

        if self.model_cfg.HEAD.CAT_TRAIN:
            self.vis_fc_merge = nn.Sequential(nn.Linear(double_embed_dim, double_embed_dim),
                                              nn.BatchNorm1d(double_embed_dim), nn.ReLU(),
                                              nn.Linear(double_embed_dim, merge_dim))
            self.lang_fc_merge = nn.Sequential(nn.LayerNorm(double_embed_dim),
                                               nn.Linear(double_embed_dim, double_embed_dim), nn.ReLU(),
                                               nn.Linear(double_embed_dim, merge_dim))

        # cls model
        if self.model_cfg.car_idloss:
            pre_shared_cls1 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls1 = nn.Sequential(*pre_shared_cls1)
            self.id_cls1 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.CAR_CLS)

        if self.model_cfg.mo_idloss:
            pre_shared_cls2 = [nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls2 = nn.Sequential(*pre_shared_cls2)
            self.id_cls2 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.MO_CLS)

        if self.model_cfg.share_idloss:
            pre_shared_cls3 = [nn.Linear(merge_dim, merge_dim), nn.BatchNorm1d(merge_dim), nn.ReLU()] \
                if self.model_cfg.HEAD.CLS_NONLINEAR else [nn.Identity()]
            self.pre_id_cls3 = nn.Sequential(*pre_shared_cls3)
            self.id_cls3 = build_softmax_cls(model_cfg=self.model_cfg, loss_type=self.model_cfg.HEAD.SHARED_CLS)

    def encode_text(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask):
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)

        if self.model_cfg.HEAD.CAT_TRAIN:
            if lang_motion_embeds.shape[0] != lang_car_embeds.shape[0]:
                lang_merge_embeds = torch.cat([lang_car_embeds.repeat(lang_motion_embeds.shape[0], 1), lang_motion_embeds], dim=-1)
            else:
                lang_merge_embeds = torch.cat(
                    [lang_car_embeds, lang_motion_embeds], dim=-1)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)

            lang_merge_embeds, lang_car_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (lang_merge_embeds, lang_car_embeds, lang_motion_embeds))
        else:
            lang_car_embeds, lang_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (lang_car_embeds, lang_motion_embeds))

            if lang_motion_embeds.shape[0] != lang_car_embeds.shape[0]:
                lang_merge_embeds = torch.cat(
                    [lang_car_embeds.repeat(lang_motion_embeds.shape[0], 1), lang_motion_embeds], dim=-1)
            else:
                lang_merge_embeds = torch.cat(
                    [lang_car_embeds, lang_motion_embeds], dim=-1)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds]

    def encode_images(self, crops, motion):
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        if self.model_cfg.HEAD.CAT_TRAIN:
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

            visual_merge_embeds, visual_car_embeds, visual_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, visual_car_embeds, visual_mo_embeds))
        else:
            visual_car_embeds, visual_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, visual_mo_embeds))
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1)

        return [visual_car_embeds, visual_mo_embeds, visual_merge_embeds]

    def forward(self, nl_mo_input_ids, nl_mo_attention_mask, nl_car_input_ids, nl_car_attention_mask,
                crops, motion, targets=None):
        # text
        outputs_mo = self.bert_model(nl_mo_input_ids, attention_mask=nl_mo_attention_mask)
        lang_motion_embeds = torch.mean(outputs_mo.last_hidden_state, dim=1)
        lang_motion_embeds = self.lang_mo_fc(lang_motion_embeds)

        outputs_car = self.bert_model(nl_car_input_ids, attention_mask=nl_car_attention_mask)
        lang_car_embeds = torch.mean(outputs_car.last_hidden_state, dim=1)
        lang_car_embeds = self.lang_car_fc(lang_car_embeds)

        # visual
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)   #batchxsize

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)

        if self.model_cfg.HEAD.CAT_TRAIN:
            lang_merge_embeds = self.lang_fc_merge(torch.cat([lang_car_embeds, lang_motion_embeds], dim=-1))
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

        # cls
        cls_logits_results = []
        if self.training and self.model_cfg.car_idloss:
            car_cls_t = self.pre_id_cls1(lang_car_embeds)
            car_cls_t = self.id_cls1(car_cls_t, targets=targets)
            cls_logits_results.append(car_cls_t)

            car_cls_v = self.pre_id_cls1(visual_car_embeds)
            car_cls_v = self.id_cls1(car_cls_v, targets=targets)
            cls_logits_results.append(car_cls_v)

        if self.training and self.model_cfg.mo_idloss:
            motion_cls_t = self.pre_id_cls2(lang_motion_embeds)
            motion_cls_t = self.id_cls2(motion_cls_t, targets=targets)
            cls_logits_results.append(motion_cls_t)

            motion_cls_v = self.pre_id_cls2(visual_mo_embeds)
            motion_cls_v = self.id_cls2(motion_cls_v, targets=targets)
            cls_logits_results.append(motion_cls_v)

        if self.training and self.model_cfg.share_idloss and self.model_cfg.HEAD.CAT_TRAIN:
            merge_cls_t = self.pre_id_cls3(lang_merge_embeds)
            merge_cls_t = self.id_cls3(merge_cls_t, targets=targets)
            cls_logits_results.append(merge_cls_t)

            merge_cls_v = self.pre_id_cls3(visual_merge_embeds)
            merge_cls_v = self.id_cls3(merge_cls_v, targets=targets)
            cls_logits_results.append(merge_cls_v)

        visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_mo_embeds = map(lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, lang_car_embeds, visual_mo_embeds, lang_motion_embeds))

        if self.model_cfg.HEAD.CAT_TRAIN:
            visual_merge_embeds, lang_merge_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, lang_merge_embeds))
        else:
            lang_merge_embeds = torch.cat([lang_car_embeds, lang_mo_embeds], dim=-1)
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1)

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds),
                (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
        }

        return out
    
