import torch
from collections import OrderedDict
from torch.backends import cudnn

from .collator import SiameseCollator, SiameseEnhancedCollator
from .siamese_baseline import SiameseBaselineModel
from .ssm import SiameseLocalandMotionModelBIG_DualTextCat
from .local_aggr import SiameseDualEnhancedEncoder
from .freeze_backbone import *


def build_model(cfg, args):
    if cfg.MODEL.NAME == "baseline":
        model = SiameseBaselineModel(cfg.MODEL)
    elif cfg.MODEL.NAME == 'dual-text-cat':
        model = SiameseCollator(cfg, SiameseLocalandMotionModelBIG_DualTextCat) # 2 bert model => 1 fc merge
    elif cfg.MODEL.NAME == 'dual-text-cat-aggr-f':
        model = SiameseEnhancedCollator(cfg, SiameseDualEnhancedEncoder)
    elif cfg.MODEL.NAME == 'dual-text-add':
        model = SiameseCollator(cfg, SiameseLocalandMotionModelBIG_DualTextCat)
    else:
        assert cfg.MODEL.NAME in ["baseline", "dual-stream", "dual-simple", "dual-stream-v2"], f"unsupported model {cfg.MODEL.NAME}"

    ossSaver = args.ossSaver

    if args.resume:
        rest_path = cfg.TEST.RESTORE_FROM
        if rest_path == "" or rest_path is None:
            rest_path = args.logs_dir + "/checkpoint_best_eval.pth"
            if cfg.DATA.USE_OSS:
                rest_path = ossSaver.get_s3_path(rest_path)

        checkpoint = ossSaver.load_ckpt(rest_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f"====> load checkpoint from {rest_path}")
    else:
        print(f"====> load checkpoint from default")

    if args.use_cuda:
        model.cuda()
        from ..executor import DataParallelExt
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = cfg.TRAIN.BENCHMARK

    return model
