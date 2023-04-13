import torch.nn as nn


def freeze_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def freeze_bn(model: nn.Module) -> None:
    def set_bn_eval(m) -> None:
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    model.apply(set_bn_eval)


def unfreeze_bn(model: nn.Module) -> None:
    def set_bn_train(m) -> None:
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model.apply(set_bn_train)


class FreezeBackbone(object):
    def __init__(self, model: nn.Module, freeze_epoch=0, tune_layers=None):
        super().__init__()
        self.model = model
        self.freeze_epoch = freeze_epoch
        self.backbone_name = tune_layers if tune_layers else ['vis_backbone', 'vis_backbone_bk']

    def start_freeze_backbone(self):
        if self.freeze_epoch <= 0:
            return
        for name in self.backbone_name:
            layer = self.model.module._modules[name]
            freeze_params(layer)
            freeze_bn(layer)
            print(f'====> Freeze {name}')

    def on_train_epoch_start(self, epoch) -> None:
        if self.freeze_epoch <= 0:
            return
        if epoch == self.freeze_epoch:
            for name in self.backbone_name:
                layer = self.model.module._modules[name]
                unfreeze_params(layer)
                unfreeze_bn(layer)
                print(f'====> Unfreeze {name}')
