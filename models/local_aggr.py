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


class SiameseDualEnhancedEncoder(torch.nn.Module):
    def __init__(self, model_cfg, data_cfg):
        super().__init__()
        print(f"====> Using visual backbone: {self._get_name()}")
        self.model_cfg = model_cfg
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        self.local_scale = nn.Parameter(torch.ones(()), requires_grad=True)
        embed_dim = self.model_cfg.EMBED_DIM
        double_embed_dim = 2 * embed_dim
        merge_dim = self.model_cfg.MERGE_DIM
        proj_dim = self.model_cfg.PROJ_DIM

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

        type_dim = len(data_cfg.TEXT_AUG_TYPE) + 1
        motion_dim = len(data_cfg.TEXT_AUG_MOTION) + 1
        color_dim = len(data_cfg.TEXT_AUG_COLOR) + 1
        size_dim = len(data_cfg.TEXT_AUG_SIZE) + 1
        
        if model_cfg.USE_NON_LINEAR_PROJ:
            print("Non-linear")
            self.vis_extract_type1 = nn.Sequential(nn.Linear(embed_dim, type_dim),
                                              nn.ReLU(), nn.Linear(type_dim, type_dim))
            self.vis_extract_type2 = nn.Sequential(nn.Linear(embed_dim, type_dim),
                                              nn.ReLU(), nn.Linear(type_dim, type_dim))
            self.vis_extract_motion1 = nn.Sequential(nn.Linear(embed_dim, motion_dim),
                                              nn.ReLU(), nn.Linear(motion_dim, motion_dim))
            # self.vis_extract_motion2 = nn.Linear(embed_dim, len(data_cfg.TEXT_AUG_MOTION) + 1)
            self.vis_extract_color1 = nn.Sequential(nn.Linear(embed_dim, color_dim),
                                              nn.ReLU(), nn.Linear(color_dim, color_dim))
            self.vis_extract_color2 = nn.Sequential(nn.Linear(embed_dim, color_dim),
                                              nn.ReLU(), nn.Linear(color_dim, color_dim))
            self.vis_extract_size1 = nn.Sequential(nn.Linear(embed_dim, size_dim),
                                              nn.ReLU(), nn.Linear(size_dim, size_dim))
            # self.vis_extract_size2 = nn.Linear(embed_dim, len(data_cfg.TEXT_AUG_SIZE) + 1)
        else:
            self.vis_extract_type1 = nn.Linear(embed_dim, type_dim)
            self.vis_extract_type2 = nn.Linear(embed_dim, type_dim)
            self.vis_extract_motion1 = nn.Linear(embed_dim, motion_dim)
            # self.vis_extract_motion2 = nn.Linear(embed_dim, len(data_cfg.TEXT_AUG_MOTION) + 1)
            self.vis_extract_color1 = nn.Linear(embed_dim, color_dim)
            self.vis_extract_color2 = nn.Linear(embed_dim, color_dim)
            self.vis_extract_size1 = nn.Linear(embed_dim, size_dim)
            # self.vis_extract_size2 = nn.Linear(embed_dim, len(data_cfg.TEXT_AUG_SIZE) + 1)
        
        self.vis_extract_inter = nn.Linear(embed_dim, 1)
        
        self.vis_proj_car = nn.Linear(embed_dim, proj_dim)
        self.vis_proj_mo = nn.Linear(embed_dim, proj_dim)
        
        total_vis_feats_dim = 2 * (len(data_cfg.TEXT_AUG_TYPE) + len(data_cfg.TEXT_AUG_COLOR) + \
                                  2 + proj_dim) + len(data_cfg.TEXT_AUG_SIZE) + 3 + len(data_cfg.TEXT_AUG_MOTION)
        # self.vis_feats_up_scale = nn.Linear(total_vis_feats_dim, double_embed_dim)
        self.lang_feats_down_scale = nn.Linear(merge_dim, total_vis_feats_dim)

        if self.model_cfg.HEAD.CAT_TRAIN:
            self.vis_fc_merge = nn.Sequential(nn.Linear(double_embed_dim, double_embed_dim),
                                              nn.BatchNorm1d(double_embed_dim), nn.ReLU(),
                                              nn.Linear(double_embed_dim, merge_dim))
            self.lang_fc_merge = nn.Sequential(nn.LayerNorm(double_embed_dim),
                                               nn.Linear(double_embed_dim, double_embed_dim), nn.ReLU(),
                                               nn.Linear(double_embed_dim, merge_dim))
            
            self.vis_fc_feats = nn.Sequential(nn.Linear(total_vis_feats_dim, total_vis_feats_dim),
                                              nn.BatchNorm1d(total_vis_feats_dim), nn.ReLU(),
                                              nn.Linear(total_vis_feats_dim, total_vis_feats_dim))

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

            lang_merge_feats = self.lang_feats_down_scale(lang_merge_embeds)

            lang_merge_embeds, lang_car_embeds, lang_mo_embeds, lang_merge_feats = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (lang_merge_embeds, lang_car_embeds, lang_motion_embeds, lang_merge_feats))
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
            lang_merge_feats = self.lang_feats_down_scale(lang_merge_embeds)

        return [lang_car_embeds, lang_mo_embeds, lang_merge_embeds, lang_merge_feats]

    def encode_images(self, crops, motion):
        visual_car_embeds = self.domian_vis_fc(self.vis_backbone(crops))
        visual_car_embeds = visual_car_embeds.view(visual_car_embeds.size(0), -1)
        
        proj_type1 = self.vis_extract_type1(visual_car_embeds)
        proj_color1 = self.vis_extract_color1(visual_car_embeds)
        proj_size1 = self.vis_extract_size1(visual_car_embeds)
        proj_car = self.vis_proj_car(visual_car_embeds)
        
        w_proj_type1 = torch.mul(self.local_scale, proj_type1)
        w_proj_color1 = torch.mul(self.local_scale, proj_color1)
        w_proj_size1 = torch.mul(self.local_scale, proj_size1)

        
        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)
        
        proj_motion1 = self.vis_extract_motion1(visual_mo_embeds)
        # proj_motion2 = self.vis_extract_motion2(visual_mo_embeds)
        proj_type2 = self.vis_extract_type2(visual_mo_embeds)
        proj_color2 = self.vis_extract_color2(visual_mo_embeds)
        # proj_size2 = self.vis_extract_size2(visual_mo_embeds)
        proj_mo = self.vis_proj_mo(visual_mo_embeds)
        
        proj_inter = self.vis_extract_inter(visual_mo_embeds)
        
        w_proj_motion1 = torch.mul(self.local_scale, proj_motion1)
        vis_feats_merge_weighted = [w_proj_type1, w_proj_motion1, w_proj_color1, w_proj_size1, proj_car,
                          proj_type2, proj_color2, proj_mo, proj_inter]
        vis_feats_merge = [proj_type1, proj_motion1, proj_color1, proj_size1, proj_car,
                          proj_type2, proj_color2, proj_mo, proj_inter]

        vis_feats_cat = torch.cat(vis_feats_merge_weighted, dim=-1)
        if self.model_cfg.HEAD.CAT_TRAIN:
            
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))

            visual_merge_feats = self.vis_fc_feats(vis_feats_cat)

            visual_merge_embeds, visual_car_embeds, visual_mo_embeds, visual_merge_feats = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, visual_car_embeds, visual_mo_embeds, visual_merge_feats))
        else:
            visual_car_embeds, visual_mo_embeds = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_car_embeds, visual_mo_embeds))
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1)
            visual_merge_feats = self.vis_fc_feats(vis_feats_cat)

        return [visual_car_embeds, visual_mo_embeds, visual_merge_embeds, visual_merge_feats]

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
        
        proj_type1 = self.vis_extract_type1(visual_car_embeds)
        proj_color1 = self.vis_extract_color1(visual_car_embeds)
        proj_size1 = self.vis_extract_size1(visual_car_embeds)
        proj_car = self.vis_proj_car(visual_car_embeds)
        
        w_proj_type1 = torch.mul(self.local_scale, proj_type1)
        w_proj_color1 = torch.mul(self.local_scale, proj_color1)
        w_proj_size1 = torch.mul(self.local_scale, proj_size1)
        

        visual_mo_embeds = self.domian_vis_fc_bk(self.vis_backbone_bk(motion))
        visual_mo_embeds = visual_mo_embeds.view(visual_mo_embeds.size(0), -1)
        
        proj_motion1 = self.vis_extract_motion1(visual_mo_embeds)
        # proj_motion2 = self.vis_extract_motion2(visual_mo_embeds)
        proj_type2 = self.vis_extract_type2(visual_mo_embeds)
        proj_color2 = self.vis_extract_color2(visual_mo_embeds)
        # proj_size2 = self.vis_extract_size2(visual_mo_embeds)
        proj_mo = self.vis_proj_mo(visual_mo_embeds)
        
        proj_inter = self.vis_extract_inter(visual_mo_embeds)
        
        w_proj_motion1 = torch.mul(self.local_scale, proj_motion1)
        vis_feats_merge_weighted = [w_proj_type1, w_proj_motion1, w_proj_color1, w_proj_size1, proj_car,
                          proj_type2, proj_color2, proj_mo, proj_inter]
        vis_feats_merge = [proj_type1, proj_motion1, proj_color1, proj_size1, proj_car,
                          proj_type2, proj_color2, proj_mo, proj_inter]

        vis_feats_cat = torch.cat(vis_feats_merge_weighted, dim=-1)

        if self.model_cfg.HEAD.CAT_TRAIN:
            lang_merge_embeds = torch.cat([lang_car_embeds, lang_motion_embeds], dim=-1)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)
            lang_merge_feats = self.lang_feats_down_scale(lang_merge_embeds)
            visual_merge_embeds = self.vis_fc_merge(torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1))
            visual_merge_feats = self.vis_fc_feats(vis_feats_cat)

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
            visual_merge_embeds, lang_merge_embeds, visual_merge_feats, lang_merge_feats = map(
                lambda t: F.normalize(t, p=2, dim=-1),
                (visual_merge_embeds, lang_merge_embeds, visual_merge_feats, lang_merge_feats))
        else:
            lang_merge_embeds = torch.cat([lang_car_embeds, lang_mo_embeds], dim=-1)
            visual_merge_embeds = torch.cat([visual_car_embeds, visual_mo_embeds], dim=-1)
            lang_merge_embeds = self.lang_fc_merge(lang_merge_embeds)
            lang_merge_feats = self.lang_feats_down_scale(lang_merge_embeds)
            visual_merge_feats = self.vis_fc_feats(vis_feats_cat)

        out = {
            "pairs": [(visual_car_embeds, lang_car_embeds), (visual_mo_embeds, lang_mo_embeds), (visual_merge_feats, lang_merge_feats),
                (visual_merge_embeds, lang_merge_embeds)],
            "logit_scale": self.logit_scale,
            "cls_logits": cls_logits_results,
            "vis_features": vis_feats_merge,
        }

        return out
    
