import random
import torch
import numpy as np
import refile
import os
import sys
import errno
import os.path as osp
from termcolor import colored
import tabulate
import copy
import argparse

from .logger import Logger
from ..configs import set_experiment


table = []
max_record = ['Max', 'Max', 0, 0, 0, 0, 0]
header = ['Method', 'Dataset', 'Epoch', 'Loss', 'MRR', 'Acc-5', 'Acc-10']
table.append(header)


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    print(f"====> set seed {seed}")

def results_record(name, dataset, epoch, losses, mrr, top5_acc, top10_acc, is_test=False):
    # result csv
    record = list()
    # name = args.name
    record.append(name)
    record.append(dataset)
    record.append(epoch)
    record.append(losses)
    record.append(mrr)
    record.append(top5_acc)
    record.append(top10_acc)
    table.append(record)
    print_table = copy.deepcopy(table)
    global max_record
    if is_test and record[-3] > max_record[-3]:
        max_record = copy.deepcopy(record)
        max_record[2] = 'Max_' + str(max_record[2])
    print_table.append(max_record)

    display = tabulate.tabulate(
        print_table,
        tablefmt="pipe",
        headers='firstrow',
        numalign="left",
        floatfmt='.3f')
    print(f"====> results in csv format: \n" + colored(display, "cyan"))


def prepare_start(arch_name):
    parser = argparse.ArgumentParser(description='AICT5 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', default="configs/single_baseline_aug1.yaml", type=str,
                        help='config_file')
    parser.add_argument('--name', default="baseline", type=str, 
                        help='experiments')
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='logs/')
    parser.add_argument('--eval_only', '-eval', action='store_true', help='only eval')
    parser.add_argument(
        "opts", 
        help="Modify config options using the command-line", 
        default=None, 
        nargs=argparse.REMAINDER, 
    )
    args = parser.parse_args()

    cfg = set_experiment(arch_name, args.config, args.opts)
    args.cfg = cfg

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    print(f"====> load config from {args.config}")

    ossSaver = MgvSaveHelper()
    ossSaver.set_stauts(save_oss=cfg.DATA.USE_OSS, oss_path=cfg.DATA.OSS_PATH)
    args.ossSaver = ossSaver

    return args, cfg


class MgvSaveHelper(object):
    def __init__(self, save_oss=False, oss_path='', echo=True):
        self.oss_path = oss_path
        self.save_oss = save_oss
        self.echo = echo

    def set_stauts(self, save_oss=False, oss_path='', echo=True):
        self.oss_path = oss_path
        self.save_oss = save_oss
        self.echo = echo

    def get_s3_path(self, path):
        if self.check_s3_path(path):
            return path
        return self.oss_path + path

    def check_s3_path(self, path):
        return path.startswith('s3:')

    def load_ckpt(self, path):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        if self.echo:
            print(f"====> load checkpoint from {path}")
        return ckpt

    def save_ckpt(self, path, epoch, model, optimizer=None):
        if self.save_oss:
            if not self.check_s3_path(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(
                    {"epoch": epoch,
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()}, f)
        else:
            torch.save(
                {"epoch": epoch,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()}, path)

        if self.echo:
            print(f"====> save checkpoint to {path}")

    def save_pth(self, path, file):
        if self.save_oss:
            if not self.check_s3_path(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(file, f)
        else:
            torch.save(file, path)

        if self.echo:
            print(f"====> save pth to {path}")

    def load_pth(self, path):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        if self.echo:
            print(f"====> load pth from {path}")
        return ckpt
