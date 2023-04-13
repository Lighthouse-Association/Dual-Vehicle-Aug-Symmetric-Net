from ..preprocessing.transforms import build_vanilla_transforms, build_transforms, build_motion_transform
import importlib


def make_data_loader(arch_name, cfg, *args, **kwargs):
    return importlib.import_module(f".{arch_name}", "Track2.dataloaders").make_data_loader(cfg, *args, **kwargs)

def prepare_data(arch_name, cfg, is_train=True):
    if not cfg.DATA.CROP_AUG:
        # CLV(1st) transforms
        transform_train = build_vanilla_transforms(cfg, is_train=True)
        transform_test = build_vanilla_transforms(cfg, is_train=False)
        motion_transform = build_vanilla_transforms(cfg, is_train=True)
    else:
        # CLV(1st) and DUN(2st)
        transform_train = build_transforms(cfg, is_train=True)
        transform_test = build_transforms(cfg, is_train=False)
        motion_transform = build_motion_transform(cfg, vanilla=True)

    return make_data_loader(arch_name, cfg, transform=transform_train, motion_transform=motion_transform, val_transform=transform_test, is_train=is_train)
