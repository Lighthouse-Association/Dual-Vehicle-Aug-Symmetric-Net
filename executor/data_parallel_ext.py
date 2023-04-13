import torch.nn as nn
from itertools import chain

class DataParallelExt(nn.DataParallel):
    """
    Temporary workaround for https://github.com/pytorch/pytorch/issues/15716.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._replicas = None
        self._outputs = None

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj,
                                                t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        self._replicas = self.replicate(self.module,
                                  self.device_ids[:len(inputs)])
        self._outputs = self.parallel_apply(self._replicas, inputs, kwargs)

        return self.gather(self._outputs, self.output_device)