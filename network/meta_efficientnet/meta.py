import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
from torchvision.models import resnet50
import math

def to_var(x, requires_grad=True):
    if torch.cuda.is_available(): x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, source_params=None,
                      solver='sgd', beta1=0.9, beta2=0.999, weight_decay=5e-4):
        if solver == 'sgd':
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src if src is not None else 0
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        elif solver == 'adam':
            for tgt, gradVal in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                exp_avg, exp_avg_sq = torch.zeros_like(param_t.data), \
                                      torch.zeros_like(param_t.data)
                bias_correction1 = 1 - beta1
                bias_correction2 = 1 - beta2
                gradVal.add_(weight_decay, param_t)
                exp_avg.mul_(beta1).add_(1 - beta1, gradVal)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, gradVal, gradVal)
                exp_avg_sq.add_(1e-8)  # to avoid possible nan in backward
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                step_size = lr_inner / bias_correction1
                newParam = param_t.addcdiv(-step_size, exp_avg, denom)
                self.set_param(self, name_t, newParam)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def setBN(self, inPart, name, param):
        if '.' in name:
            part = name.split('.')
            self.setBN(getattr(inPart, part[0]), '.'.join(part[1:]), param)
        else:
            setattr(inPart, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copyModel(self, newModel, same_var=False):
        # copy meta model to meta model
        tarName = list(map(lambda v: v, newModel.state_dict().keys()))
        # requires_grad
        partName, partW = list(map(lambda v: v[0], newModel.named_params(newModel))), list(
            map(lambda v: v[1], newModel.named_params(newModel)))  # new model's weight
        metaName, metaW = list(map(lambda v: v[0], self.named_params(self))), list(
            map(lambda v: v[1], self.named_params(self)))
        # bn running_vars
        bnNames = list(set(tarName) - set(partName))
        # copy vars
        for name, param in zip(metaName, partW):
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(self, name, param)
        # copy training mean var
        tarName = newModel.state_dict()
        for name in bnNames:
            param = to_var(tarName[name], requires_grad=False)
            self.setBN(self, name, param)

    def copyWeight(self, modelW):
        # copy state_dict to buffers
        curName = list(map(lambda v: v[0], self.named_params(self)))
        tarNames = set()
        for name in modelW.keys():
            if name.startswith("module"):
                tarNames.add(".".join(name.split(".")[1:]))
            else:
                tarNames.add(name)
        bnNames = list(tarNames - set(curName))
        for tgt in self.named_params(self):
            name_t, param_t = tgt
            try:
                param = to_var(modelW[name_t], requires_grad=True)
            except:
                param = to_var(modelW['module.' + name_t], requires_grad=True)
            self.set_param(self, name_t, param)
        for name in bnNames:
            try:
                param = to_var(modelW[name], requires_grad=False)
            except:
                param = to_var(modelW['module.' + name], requires_grad=False)
            self.setBN(self, name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.in_features = args[0]
        self.out_features = args[1]

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True) if ignore.bias is not None else None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        #output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  0, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        val2 = self.weight.sum()
        res = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                           self.training or not self.track_running_stats, self.momentum, self.eps)
        return res

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
