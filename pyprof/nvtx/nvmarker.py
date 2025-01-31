#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file intercepts (monkey patches) the following functions and adds NVTX markers.
	torch.*
	torch.Tensor.*
	torch.nn.functional.*
	torch.nn.*.forward

The NVTX markers (one or more) contain the following information
	call trace (a list of file_name:line_number)
	extra_repr() from torch.nn modules
	module/class name
	function name
	inputs (args and kwargs)
		scalar: name, type and value
		tensor: name, shape and datatype
		numpy: name, shape and datatype
		list/tuple: a sequence of scalars or tensors or numpy arrays
"""

import torch
import torch.cuda.nvtx as nvtx
import numpy
import inspect as ins
import traceback
import math
import json
import importlib



def isfunc(mod, f):
    assert hasattr(mod, f)
    attr = getattr(mod, f)

    # Ignore functions like _add
    if (len(f) >= 2):
        if f[0] == "_" and f[1] != "_":
            return False

    # Ignore functions from this list
    ignore = [
        '__all__', '__array__', '__array_priority__', '__array_wrap__', '__bool__', '__builtins__', '__cached__',
        '__class__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__doc__', '__file__',
        '__format__', '__getattribute__', '__getitem__', '__hash__', '__index__', '__init__', '__init_subclass__',
        '__iter__', '__len__', '__loader__', '__module__', '__name__', '__new__', '__nonzero__', '__package__',
        '__path__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__setattr__', '__setitem__',
        '__setstate__', '__sizeof__', '__spec__', '__str__', '__subclasshook__', '__version__', '__weakref__'
    ]

    # Add functions to this list if they cause recursion
    ignore += ['size', 'tolist', 'dim', 'is_storage', 'item', 'data_ptr']

    # Add functions to this list if they are called often, are generally extremely
    # short, and don't lead GPU usage
    #
    ignore += [
        'autocast_decrement_nesting', 'autocast_increment_nesting', 'contiguous', 'detach', 'empty', 'from_numpy',
        'has_torch_function', 'is_autocast_enabled', 'is_available', 'is_complex', 'is_floating_point',
        'is_grad_enabled', 'is_initialized', 'is_tensor', 'numel', 'requires_grad_', 'set_autocast_enabled', 'to',
        'type'
    ]

    if f in ignore:
        return False
    return ins.ismethod(attr) or ins.isfunction(attr) or ins.ismethoddescriptor(attr) or ins.isbuiltin(attr)


# Returns a dict string with a tracemarker and function stack in it
#
def traceMarker(stack):
    d = {}
    cadena = []
    for i in range(len(stack) - 1):
        fi = stack[i]
        t = f"{fi.filename}:{fi.lineno}"
        cadena.append(t)
    d['traceMarker'] = cadena
    return str(d)


def modMarker(mod, fn_name, args):
    """
	Returns the stringified extra_repr() of a module.
	"""
    assert (len(args) > 0)
    d = {}
    d['mod'] = mod.__name__
    d['strRepr'] = args[0].extra_repr()
    return str(d)


def add_wrapper(mod, fn_name):
    # Get a pointer to the original function
    func = getattr(mod, fn_name)

    # Check if the mod has a string representation
    # and is not a Script or Traced module (used by JIT)
    # yapf: disable
    s = hasattr(mod, "extra_repr") and (type(mod) is not torch.jit.ScriptModule
                                        ) and (type(mod) is not torch.jit.TopLevelTracedModule)

    # yapf: enable
    # print(f'wrap {mod.__name__}:{fn_name}')
    def wrapper_func(*args, **kwargs):

        # Extract the stacktrace
        stack = traceback.extract_stack()

        # Push trace marker
        nvtx.range_push(traceMarker(stack))

        # Push module marker
        if s:
            m = modMarker(mod, fn_name, args)
            nvtx.range_push(m)

        # Create and push argument marker
        cadena = argMarker(mod, fn_name, args, kwargs)
        nvtx.range_push(cadena)

        # # Create and push layer marker
        # info = 'layer:' +  mod.__name__ + ',' + fn_name
        # nvtx.range_push(info)

        # Call the original function
        result = func(*args, **kwargs)

        # # pop layer marker
        # nvtx.range_pop()

        # Pop argumet marker
        nvtx.range_pop()

        # Pop module marker
        if s:
            nvtx.range_pop()

        # Pop trace marker
        nvtx.range_pop()

        return result

    setattr(mod, fn_name, wrapper_func)


def argMarker(mod, op, args, kwargs):
    # For this function args is a tuple and kwargs is a dict

    def tensor(arg, name=""):
        a = {}
        a['name'] = name
        a['type'] = "tensor"
        a['shape'] = tuple(arg.size())
        a['dtype'] = str(arg.dtype).split(".")[-1]
        cadena['args'].append(a)

    def ndarray(arg, name=""):
        a = {}
        a['name'] = name
        a['type'] = "ndarray"
        a['shape'] = arg.shape
        a['dtype'] = str(arg.dtype).split(".")[-1]
        cadena['args'].append(a)

    def seq(arg, name=""):
        assert issequence(arg)
        a = {}
        a['name'] = name
        if isinstance(arg, list):
            a['type'] = "list"
            a['value'] = arg
        else:
            a['type'] = "tuple"
            # The arg could be torch.Size, which is a subclass of tuple
            # Therefore, explicitly convert to tuple
            a['value'] = tuple(arg)

        cadena['args'].append(a)

    def scalar(arg, name=""):
        a = {}
        a['name'] = name
        a['type'] = type(arg).__name__
        # handle the case when the argument is +/- inf or nan
        if arg == float('inf'):
            a['value'] = "inf"
        elif arg == float('-inf'):
            a['value'] = "-inf"
        elif isinstance(arg, float) and math.isnan(arg):
            a['value'] = "nan"
        else:
            a['value'] = arg
        cadena['args'].append(a)

    def isscalar(arg):
        return (type(arg) is int) or (type(arg) is float) or (type(arg) is bool) or (arg is None) or (type(arg) is str)

    def issequence(arg):
        return isinstance(arg, list) or isinstance(arg, tuple)

    def foo(args, name):
        # args should be an iterable sequence e.g. list or tuple
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.dim() == 0:
                    scalar(arg.item(), name)
                else:
                    tensor(arg, name)
            elif isinstance(arg, numpy.ndarray):
                ndarray(arg, name)
            elif (isscalar(arg)):
                scalar(arg, name)
            elif issequence(arg):
                if (len(arg) == 0) or isscalar(arg[0]):  # An empty sequence or a sequence of scalars
                    seq(arg, name)
                else:  # A sequence of tensors or numpy arrays
                    foo(arg, name)
            '''
			else:
				print("The following arg is none of Tensor, numpy array, scalar but a %s" % (str(type(arg))))
				print("Mod: %s" % str(mod.__name__))
				print("Op: %s" % str(op))
				print(dir(arg))
			'''

    cadena = {}
    cadena['mod'] = mod.__name__
    cadena['op'] = op
    cadena['args'] = []

    foo(args, "")
    for k, v in kwargs.items():
        foo((v,), k)

    return str(cadena)


def patchClass(cls):
    for f in dir(cls):
        if isfunc(cls, f):
            add_wrapper(cls, f)


def patch_torch_classes():
    """Monkey-patch all classes in torch"""
    for cls in [torch, torch.Tensor, torch.nn.functional, torch.distributed]:
        patchClass(cls)


def patch_torch_nn_forward_functions():
    """Monkey-patch all forward functions in torch.nn libraries"""
    for cls in [torch.nn.RNN, torch.nn.RNNCell, torch.nn.LSTM, torch.nn.LSTMCell, torch.nn.GRU, torch.nn.GRUCell]:
        if isfunc(cls, 'forward'):
            add_wrapper(cls, 'forward')


def patch_dataloader():
    """Monkey-patch the dataloader in torch.utils.data"""
    mod = torch.utils.data.dataloader
    old_iter = mod.DataLoader.__iter__

    def new_iter(self, *args, **kwargs):
        # Push trace marker
        nvtx.range_push(traceMarker("Dataloader"))

        # First pass is for creating the dataloader + returning the first data
        cadena = argMarker(mod, "DataLoader", args, kwargs)
        nvtx.range_push(cadena)

        for x in old_iter(self, *args, **kwargs):
            # Pop tracemarker
            nvtx.range_pop()

            # Dataloader stop, Model start
            nvtx.range_pop()

            yield x

            # Push trace marker
            nvtx.range_push(traceMarker("DataLoader"))

            # Model stop, dataloader start
            cadena = argMarker(mod, "DataLoader", args, kwargs)
            nvtx.range_push(cadena)

        # Pop the last iteration before returning
        nvtx.range_pop()
        nvtx.range_pop()

    mod.DataLoader.__iter__ = new_iter


def patch_apex():
    """Monkey-patch functions in APEX"""
    patch_apex_c()
    patch_apex_pyt()


def patch_apex_c():
    """Monkey-patch C-implemented functions in APEX"""
    if importlib.util.find_spec("amp_C") is not None:
        import amp_C
        patchClass(amp_C)

    if importlib.util.find_spec("fused_adam_cuda") is not None:
        import fused_adam_cuda
        patchClass(fused_adam_cuda)

    if importlib.util.find_spec("fused_lamb_cuda") is not None:
        import fused_lamb_cuda
        patchClass(fused_lamb_cuda)

    if importlib.util.find_spec("fused_layer_norm_cuda") is not None:
        import fused_layer_norm_cuda
        patchClass(fused_layer_norm_cuda)

    if importlib.util.find_spec("distributed_lamb_cuda") is not None:
        import distributed_lamb_cuda
        patchClass(distributed_lamb_cuda)

    if importlib.util.find_spec("xentropy_cuda") is not None:
        import xentropy_cuda
        patchClass(xentropy_cuda)

    if importlib.util.find_spec("mlp_cuda") is not None:
        import mlp_cuda
        patchClass(mlp_cuda)


def patch_apex_pyt():
    """Monkey-patch pytorch-implemented functions in APEX"""
    if importlib.util.find_spec("apex") is not None:
        patch_apex_module("apex.amp")
        patch_apex_module("apex.contrib.groupbn")
        patch_apex_module("apex.contrib.multihead_attn")
        patch_apex_module("apex.contrib.optimizers")
        patch_apex_module("apex.contrib.sparsity")
        patch_apex_module("apex.contrib.xentropy")
        patch_apex_module("apex.fp16_utils")
        patch_apex_module("apex.mlp")
        patch_apex_module("apex.multi_tensor_apply")
        patch_apex_module("apex.optimizers")
        patch_apex_module("apex.parallel")
        # patch_apex_module("apex.RNN")  # Confirmed to be dead code. Do not patch


def is_same_module_or_submodule(orig, incoming):
    """
    Returns true if the incoming module is the same module as the original, 
    or is a submodule of the original module
    """
    if incoming is None:
        return False
    if orig == incoming:
        return True
    if incoming.__name__.startswith(orig.__name__):
        return True
    return False


def patch_apex_module(modstr):
    """
    Patch all forward/backward/step functions in classes in the given apex module.
    """
    if importlib.util.find_spec(modstr) is not None:
        mod = importlib.import_module(modstr)

        for _, v in ins.getmembers(mod):
            # This makes sure we don't patch random other modules that are imported by the target module
            #
            if is_same_module_or_submodule(mod, ins.getmodule(v)):
                if (ins.isclass(v)):
                    patch_apex_class(v)


def patch_apex_class(cls):
    """
    Patch all forward/backward/step functions in the given apex class
    """
    for f in cls.__dict__:
        if (ins.isfunction(cls.__dict__[f])):
            if f in ["forward", "backward", "step"]:
                add_wrapper(cls, f)


def init(*args, **kwargs):
    """
    Initialize pyprof and monkey-patch Torch functions
    """

    print("Initializing NVTX monkey patches")

    patch_apex()
    # patch_dataloader()
    patch_torch_classes()
    # 找到所有的module,修改其中的forward函数，即对forward进行封装，在其前后设置nvmarker，用于监测
    patch_torch_nn_forward_functions()
    print("Done with NVTX monkey patching")
