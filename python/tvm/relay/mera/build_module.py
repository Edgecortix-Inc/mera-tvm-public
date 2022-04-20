# Copyright 2022 EdgeCortix Inc.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import logging

from ... import get_global_func, runtime
from .. import transform
from .. import expr as _expr
from ..param_dict import save_param_dict
from ..op.contrib.register import get_pattern_table
from ..build_module import bind_params_by_name
from ..build_module import build as _build
from ...ir.transform import Sequential, PassContext
from ...contrib import graph_executor, cc


class BuildConfig(object):
    current = None

    default_arch_config = {
        "arch": "DNAF200L0002",
    }

    default_compiler_config = {
        "max_tile_height": 64,
        "max_tile_width": 64,
        "dump_ir": "true",
        "use_small_acc_mem": "true",
        "max_acc_tile_height": 16,
        "max_acc_tile_width": 32,
        "target": None,
        "compiler_workers": 7,
        "sim_freq_mhz" : 800,
        "scheduler_config": {
            "mode": "Fast",
            "pre_scheduling_iterations": 8000,
            "main_scheduling_iterations": 32000,
        }
    }

    def __init__(self, **kwargs):
        self.arch_config = BuildConfig.default_arch_config.copy()
        self.compiler_config = BuildConfig.default_compiler_config.copy()
        self._old_scope = None

        for k, v in kwargs.items():
            if k in self.arch_config:
                self.arch_config[k] = v
            elif k in self.compiler_config:
                self.compiler_config[k] = v

    def __getattr__(self, name):
        if name in self.arch_config:
            self.arch_config[name]
        elif name in self.compiler_config:
            self.compiler_config[name]
        else:
            raise ValueError("Unknown config param %s" % name)

    def __enter__(self):
        self._old_scope = BuildConfig.current
        acfg = BuildConfig.current.arch_config.copy()
        ccfg = BuildConfig.current.compiler_config.copy()
        acfg.update(self.arch_config)
        ccfg.update(self.compiler_config)
        self.arch_config = acfg
        self.compiler_config = ccfg
        BuildConfig.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope
        BuildConfig.current = self._old_scope


BuildConfig.current = BuildConfig()


def build_config(**kwargs):
    """
    Set the configuration parameters required by each target

    The only required parameter is "target". It should be one of
    "Interpreter", "InterpreterHw", "Simulator", "IP".

    The rest of parameters that can be configured are listed in
    BuildConfig above.

    Example usage:

    cfg = {"arch": 200}
    with mera.build_config(target="Interpreter", **cfg):
       mera.build(mod, params, host_arch="x86", output_dir=output_dir)

    Returns
    -------
    config: BuildConfig
    """
    if "target" not in kwargs:
        raise ValueError("target is not provided")

    return BuildConfig(**kwargs)


def _set_config(cfg, use_interpreter):
    def _get_config_string(config, indent=0):
        ret = ""
        for k, v in config.items():
            if not isinstance(v, dict):
                ret += " " * indent + k + ": " + str(v) + "\n"
            else:
                ret += " " * indent + k + ": " + "\n"
                ret += _get_config_string(v, indent + 2)

        return ret.rstrip()

    compiler_config_str = _get_config_string(cfg.compiler_config)
    logging.debug(f"Mera Compiler Config =\n{compiler_config_str}")
    mera_set_ccfg = get_global_func("relay.ext.mera.set_ccfg")
    mera_set_ccfg(compiler_config_str)

    if not use_interpreter:
        arch_cfg = cfg.arch_config
        if isinstance(arch_cfg["arch"], dict):
            # We are overriding arch config with custom values, serialize the contents directly so loader works correctly
            arch_cfg = arch_cfg["arch"]
        arch_config_str = _get_config_string(arch_cfg)
        mera_set_arch = get_global_func("relay.ext.mera.set_arch")
        mera_set_arch(arch_config_str)


def _build_common(mod, params, target_tvm, fcompile, layout, output_dir, aux_config):
    assert get_global_func("relay.ext.mera")

    cfg = BuildConfig.current
    assert cfg.compiler_config["target"] is not None

    use_interpreter = cfg.compiler_config["target"] in ["Interpreter", "InterpreterHw"]

    _set_config(cfg, use_interpreter)

    if aux_config["with_clip_pattern"]:
        pattern_table = get_pattern_table("mera_with_clip")
    else:
        pattern_table = get_pattern_table("mera")

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    with PassContext(opt_level=3):
        if "insert_downsample" in aux_config:
            msg = "only NHWC layout supported for inserting downsampling"
            assert layout == "NHWC", msg
            new_in_height, new_in_width = aux_config["insert_downsample"]
            mod = transform.InsertResize(new_in_height, new_in_width)(mod)

        if layout == "NCHW":
            # swap pad and layout transform because TVM inserts layout transform
            # between pad and qconv
            # also run EliminateCommonSubexpr because layout transform can be duplicated
            # for residual block
            def fskip(expr):
                if isinstance(expr, _expr.Call) and expr.op.name != "layout_transform":
                    return True
                return False

            desired_layout = {"qnn.conv2d": ["NHWC", "default"]}
            layout_transform = Sequential(
                [
                    transform.ConvertLayout(desired_layout),
                    transform.SwapPadLayoutTransform(),
                    transform.EliminateCommonSubexpr(fskip),
                ]
            )
            mod = layout_transform(mod)

        pass_list = [
            transform.SimplifyInference(),
            transform.MergeComposite(pattern_table),
            transform.AnnotateTarget("mera"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InlineExternClip(),
            transform.RemoveUnusedFunctions(),
        ]

        if aux_config["prequantize_input"]:
            pass_list.append(transform.RemoveInputQuantize())

        composite_partition = Sequential(pass_list)

        partitioned = composite_partition(mod)

    input_layout = "NHWC"
    config = {
        "relay.ext.mera.options": {
            "input_layout": input_layout,
            "weight_layout": aux_config["weight_layout"],
        }
    }

    with PassContext(opt_level=3, disabled_pass=["AlterOpLayout"], config=config):
        json, lib, all_params = _build(partitioned, target=target_tvm, params=params)

    # remove constants that are embedded in Mera IR
    params = {}
    for k, v in all_params.items():
        if not str(k).startswith("mera"):
            params[k] = v

    lib_path = os.path.join(output_dir, "deploy.so")
    with open(os.path.join(output_dir, "deploy.json"), "w") as f:
        f.write(json)
    with open(os.path.join(output_dir, "deploy.params"), "wb") as f:
        f.write(save_param_dict(params))

    lib.export_library(lib_path, fcompile=fcompile)

    # clear arch for later execution
    mera_set_arch = get_global_func("relay.ext.mera.set_arch")
    mera_set_arch("")

    return json, params, lib_path


def build(mod, params, host_arch="x86", output_dir=None, layout="NCHW", aux_config={}):
    """The function to compile modules for Mera target with x86 or ARM hosts.
    Parameters
    ----------
    mod : tvm.IRModule
        A Relay module to build. This should be the output of frontends.
        The model must be in the NCHW layout.

    params : dict of str to NDArray
        Input parameters to the graph. They are another outputs of frontends.

    host_arch : str, "x86" or "arm"
        If this is "x86", the compiled module would run on the interpreter
        or the simulator (specified via BuildConfig).
        Otherwise, the module will be cross compiled to run on ARM hosts.

    output_dir : str
        An output directory where a compiled library will be exported.
        If None, a directory named "_out" will be created in the current directory.

    layout: str
        The layout of an input tensor, either "NCHW" or "NHWC".
        It must be the same layout as the output of frontends expects.

    Returns
    -------
    json : str
       The output graph serialized to a json string
    params : dict of str to NDArray
       Input parameters to the runtime graph module
    lib_path : str
       The path to a compiled shared library
    """

    if host_arch == "x86":
        target = "llvm -mcpu=core-avx2"
        fcompile = cc.create_shared
    else:
        assert host_arch == "arm"
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
        fcompile = cc.cross_compiler("aarch64-linux-gnu-g++")

    if output_dir is None:
        output_dir = "_out"

    os.makedirs(output_dir, exist_ok=True)

    if "weight_layout" not in aux_config:
        aux_config["weight_layout"] = "OIHW"

    if "prequantize_input" not in aux_config:
        aux_config["prequantize_input"] = False

    if "with_clip_pattern" not in aux_config:
        aux_config["with_clip_pattern"] = False

    return _build_common(
        mod, params, target, fcompile, layout=layout, output_dir=output_dir, aux_config=aux_config
    )


def build_for_tflite(mod, params, host_arch="x86", output_dir=None):
    """A specialization for mera.build(...) function above for tflite"""
    return build(
        mod,
        params,
        host_arch=host_arch,
        output_dir=output_dir,
        layout="NHWC",
        aux_config={"weight_layout": "HWIO"},
    )


def load_runtime_module(json, params, lib_path):
    """A convienience function to create a runtime module from compiled outputs
    It can be used only on x86 hosts.
    Parameters
    ----------
    json : str
       The compiled graph serialized to a json string
    params : dict of str to NDArray
       Input parameters to the runtime graph module
    lib_path : str
       The path to a compiled shared library

    Returns
    -------
    rt_mod : GraphModule
        A runtime graph module ready to execute.
    """
    lib = runtime.load_module(lib_path)
    ctx = runtime.cpu()
    rt_mod = graph_executor.create(json, lib, ctx)
    rt_mod.set_input(**params)

    return rt_mod
