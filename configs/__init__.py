import os
import importlib

def set_experiment(arch_name, exp_name, opts=None):
    dirname = os.path.dirname(__file__)
    filePath = os.path.join(dirname, f'{arch_name}/{exp_name}.yaml')
    cfg = importlib.import_module(f".{arch_name}.config", "Track2.configs").get_default_config()
    print(filePath)
    cfg.merge_from_file(filePath)
    if opts:
        cfg.merge_from_list(opts)
    # cfg.freeze()
    return cfg
