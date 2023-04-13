import os

def set_experiment(arch_name, exp_name, opts=None):
    dirname = os.path.dirname(__file__)
    filePath = os.path.join(dirname, f'{arch_name}/{exp_name}.yaml')
    cfg = __import__(f".{arch_name}.config").get_default_config()
    cfg.merge_from_file(filePath)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg
