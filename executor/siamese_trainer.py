import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from ..benchmark import Timer, AverageMeter, ProgressMeter, accuracy, get_mrr, Counter
from ..models import FreezeBackbone
from ..optimizer import build_vanilla_optimizer
from .utils import build_tokenizer
from ..evaluate import evaluate_by_test_all, evaluate_by_test


class SiameseTrainer:

    def __init__(self, cfg, 
                 model: nn.DataParallel,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 freeze_model = True,
                 optimizer = None,
                 scheduler = None,
                 tokenizer = None,
                 save_interval = 200,
                 arch = "ssm"):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_freeze = None
        if freeze_model:
            self.model_freeze = FreezeBackbone(self.model, freeze_epoch=self.cfg.TRAIN.FREEZE_EPOCH)
        self.optimizer = optimizer
        self.scheduler = scheduler
        if not self.optimizer or not self.scheduler:
            self.optimizer, self.scheduler = build_vanilla_optimizer(self.cfg, self.model, self.train_loader)
        self.tokenizer = tokenizer if tokenizer else build_tokenizer(self.cfg)
        self.save_interval = save_interval
        self.arch = arch

    def train_iteration(self, iteration, args, global_step: Counter, best_mrr: Counter):
        self.model.train()

        inner_meters = {}
        build_meters = getattr(self.model.module, "build_meters", None)
        if callable(build_meters):
            inner_meters = build_meters()
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        mrr = AverageMeter('MRR', ':6.4f')
        top10_acc = AverageMeter('Acc@10', ':6.4f')
        top5_acc = AverageMeter('Acc@5', ':6.4f')
        progress = ProgressMeter(
            len(self.train_loader)*self.cfg.TRAIN.ONE_EPOCH_REPEAT, 
            [batch_time, data_time, losses, *inner_meters.values(), mrr, top5_acc, top10_acc],
            prefix="Epoch: [{}]".format(iteration))
        epo_rep_timer = Timer()
        epo_timer = Timer()

        for tmp in range(self.cfg.TRAIN.ONE_EPOCH_REPEAT):
            self.model_freeze.on_train_epoch_start(epoch=iteration * self.cfg.TRAIN.ONE_EPOCH_REPEAT + tmp)
            for batch_idx, batch in enumerate(self.train_loader):
                image, text, car_text, view_text = batch["crop_data"], batch["text"], batch["car_text"], batch["view_text"]
                id_car, cam_id = batch["tmp_index"], batch["camera_id"]

                # same dual Text
                if self.cfg.MODEL.SAME_TEXT:
                    car_text = text

                tokens = self.tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
                data_time.update(epo_rep_timer.update())
                global_step.increment()
                self.optimizer.zero_grad()

                added_params = [id_car, inner_meters]
                if self.arch == "local_aggr":
                    g_feats = batch["type1"], batch["type2"], batch["motion1"], \
                        batch["color1"], batch["color2"], batch["size1"], batch["intersection"]
                    added_params.append(g_feats)


                if self.cfg.DATA.USE_MOTION:
                    bk = batch["bk_data"]
                    if 'dual-text' in self.cfg.MODEL.NAME:
                        car_tokens = self.tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
                        if 'view' in self.cfg.MODEL.NAME:
                            view_tokens = self.tokenizer.batch_encode_plus(view_text, padding='longest', return_tensors='pt')
                            outputs = self.model(*added_params, tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                            car_tokens['input_ids'].cuda(), car_tokens['attention_mask'].cuda(),
                                            view_tokens['input_ids'].cuda(), view_tokens['attention_mask'].cuda(),
                                            image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                        else:
                            # with out nlp view
                            outputs = self.model(*added_params, tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                            car_tokens['input_ids'].cuda(), car_tokens['attention_mask'].cuda(),
                                            image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                    else:
                        # without dual text
                        if 'view' in self.cfg.MODEL.NAME:
                            view_tokens = self.tokenizer.batch_encode_plus(view_text, padding='longest', return_tensors='pt')
                            outputs = self.model(*added_params, tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                            view_tokens['input_ids'].cuda(), view_tokens['attention_mask'].cuda(),
                                            image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                        else:
                            outputs = self.model(*added_params, tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                            image.cuda(), bk.cuda(), targets=id_car.long().cuda())
                else:
                    # without motion
                    outputs = self.model(*added_params, tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                    image.cuda(), targets=id_car.long().cuda())
                loss, acc_sim = outputs
                
                # get_acc_sim = getattr(self.model.module, "get_acc_sim", None)
                # acc_sim = get_acc_sim()

                acc5, acc10 = accuracy(acc_sim, torch.arange(image.size(0)).cuda(), topk=(5, 10))
                mrr_ = get_mrr(acc_sim)

                losses.update(loss.item(), image.size(0))
                mrr.update(mrr_.item() * 100, image.size(0))
                top10_acc.update(acc10[0], image.size(0))
                top5_acc.update(acc5[0], image.size(0))
                
                wandb.log({
                    "loss_total": losses.avg,
                    "mrr": mrr.avg,
                    "Acc@10": top10_acc.avg,
                    "Acc@5": top5_acc.avg
                })

                loss.backward()
                self.optimizer.step()

                self.scheduler.step()
                batch_time.update(epo_rep_timer.update())
                epo_rep_timer.reset()

                if batch_idx % self.cfg.TRAIN.PRINT_FREQ == 0:
                    progress.display(global_step.count % (len(self.train_loader) * 30))
        
        if iteration % self.save_interval == 1:
            checkpoint_file = args.logs_dir + "/checkpoint_%d.pth" % iteration
            args.ossSaver.save_ckpt(checkpoint_file, iteration, self.model, self.optimizer)

        if mrr.avg > best_mrr.count:
            best_mrr.update(mrr.avg)
            checkpoint_file = args.logs_dir + "/checkpoint_best.pth"
            args.ossSaver.save_ckpt(checkpoint_file, iteration, self.model, self.optimizer)

        print(f'Epoch {iteration} running time: ', epo_timer)

    def train(self, args, end_evaluate=True):
        train_timer = Timer()
        self.model.train()
        global_step = Counter()
        best_mrr = Counter()

        self.model_freeze.start_freeze_backbone()
        for epoch in range(self.cfg.TRAIN.EPOCH):
            if self.cfg.EVAL.EVAL_BY_TEST and (epoch + 1) % (self.cfg.EVAL.EPOCH * self.cfg.EVAL.EVAL_BY_TEST_NUM) == 0:
                evaluate_by_test(self.model, self.val_loader, epoch, self.cfg, args.feat_idx, args, self.tokenizer, self.optimizer)
            self.train_iteration(epoch, args, global_step, best_mrr)

        wandb.finish()
        if end_evaluate:
            evaluate_by_test_all(self.model, self.val_loader, self.cfg.TRAIN.EPOCH, self.cfg, args.feat_idx, args, self.tokenizer, self.optimizer)
        print('Total running time: ', train_timer)
        print(f'Logs dir: {args.logs_dir} ')
