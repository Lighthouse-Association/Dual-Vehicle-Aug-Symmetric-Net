import torch
from collections import OrderedDict
from torch.backends import cudnn

from .collator import SiameseCollator
from .siamese_baseline import SiameseBaselineModel
from .ssm import SiameseLocalandMotionModelBIG_DualTextCat
from .freeze_backbone import *


def build_model(cfg, args):
    if cfg.MODEL.NAME == "baseline":
        model = SiameseBaselineModel(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-text-cat':
        model = SiameseCollator(cfg, SiameseLocalandMotionModelBIG_DualTextCat) # 2 bert model => 1 fc merge
    elif cfg.MODEL.NAME == 'dual-text-add':
        model = SiameseCollator(cfg, SiameseLocalandMotionModelBIG_DualTextCat)
    else:
        assert cfg.MODEL.NAME in ["baseline", "dual-stream", "dual-simple", "dual-stream-v2"], f"unsupported model {cfg.MODEL.NAME}"

    ossSaver = args.ossSaver

    if args.resume:
        if cfg.TEST.RESTORE_FROM == "" or cfg.TEST.RESTORE_FROM is None:
            cfg.TEST.RESTORE_FROM = args.logs_dir + "/checkpoint_best_eval.pth"
            if cfg.DATA.USE_OSS:
                cfg.TEST.RESTORE_FROM = ossSaver.get_s3_path(cfg.TEST.RESTORE_FROM)

        checkpoint = ossSaver.load_ckpt(cfg.TEST.RESTORE_FROM)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print(f"====> load checkpoint from default")

    if args.use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = cfg.TRAIN.BENCHMARK

    return model
