import os

from .executor import SiameseTrainer
from .evaluate import evaluate_by_test_all
from .utils import set_seed, prepare_start
from .dataloaders import prepare_data
from .models import build_model


def main():
    arch_name = "ssm"
    args, cfg = prepare_start(arch_name)
    if cfg.MODEL.METRIC.LOSS == '':
        set_seed(cfg.TRAIN.SEED, cfg.TRAIN.DETERMINISTIC)
    os.makedirs(args.logs_dir, exist_ok=True)
    # print(cfg)

    train_data, train_loader, _, val_loader = prepare_data(arch_name, cfg)

    args.use_cuda = True
    if cfg.MODEL.NUM_CLASS == 0:
        cfg.MODEL.NUM_CLASS = len(train_data)

    if args.eval_only:
        args.resume = True
    model = build_model(cfg, args)
    trainer = SiameseTrainer(cfg, model, train_loader, val_loader)

    args.feat_idx = cfg.MODEL.MAIN_FEAT_IDX
    if args.eval_only:
        evaluate_by_test_all(model, val_loader, 0, cfg, args.feat_idx, args, trainer.tokenizer, trainer.optimizer)
        return
    
    trainer.train(args)

if __name__ == '__main__':
    main()

