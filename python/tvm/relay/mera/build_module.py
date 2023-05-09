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
import sys
import numpy as np

from ... import get_global_func, runtime
from ... import tir
from ... import IRModule
from .. import transform
from ..function import Function
from .. import expr as _expr
from ..param_dict import save_param_dict
from ..op.contrib.register import get_pattern_table
from ..build_module import bind_params_by_name
from ..build_module import build as _build
from ..expr import GlobalVar, bind
from ..expr_functor import ExprMutator, ExprVisitor
from ...ir.transform import Sequential, PassContext
from ...contrib import graph_executor, cc
from ..._ffi.base import TVMError, register_error

logger = logging.getLogger('mera-tvm')

@register_error
class MeraError(TVMError):
    """Base error found during compilation of a MERA model. """

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
        "manual_sg_merge_map" : "",
        "use_legacy_sg_cutting": "false",
        "batch_cutting_factor": 1,
        "scheduler_config": {
            "mode": "Fast",
            "pre_scheduling_iterations": 8000,
            "main_scheduling_iterations": 32000,
            "batch_interleave" : 0,
            "shared_data_mode" : "false",
            "shared_weight_mode" : "false",
            "wide_input_mode" : "false",
            "wide_kernel_mode" : "false",
            "progress_bars" : "false",
            "consider_allocation" : "false",
            "partitions" : 1
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


def parse_config(cfg):
    def _get_config_string(config, indent=0):
        ret = ""
        for k, v in config.items():
            if not isinstance(v, dict):
                v_str = str(v)
                if isinstance(v, str) and v == "":
                    v_str = "''"
                ret += " " * indent + k + ": " + v_str + "\n"
            else:
                ret += " " * indent + k + ": " + "\n"
                ret += _get_config_string(v, indent + 2)

        return ret.rstrip()
    compiler_config_str = _get_config_string(cfg.compiler_config)
    arch_cfg = cfg.arch_config
    if isinstance(arch_cfg["arch"], dict):
        # We are overriding arch config with custom values, serialize the contents directly so loader works correctly
        arch_cfg = arch_cfg["arch"]
    arch_config_str = _get_config_string(arch_cfg)
    return compiler_config_str, arch_config_str


def _set_config(cfg):
    compiler_config_str, arch_config_str = parse_config(cfg)
    mera_set_ccfg = get_global_func("relay.ext.mera.set_ccfg")
    mera_set_ccfg(compiler_config_str)
    mera_set_arch = get_global_func("relay.ext.mera.set_arch")
    mera_set_arch(arch_config_str)
    return compiler_config_str, arch_config_str


class IsComputeIntensiveGraph(ExprVisitor):
    """Visit the graph recursively and check if it contains compute heavy ops."""

    def __init__(self):
        ExprVisitor.__init__(self)
        self.is_compute_intensive = False

    def visit_call(self, call):
        compute_intensive_ops = set(
            [
                "qnn.conv2d",
                "qnn.dense",
                "nn.max_pool2d",
                "nn.avg_pool2d",
                "nn.adaptive_avg_pool2d",
                "image.resize2d",
                "mean",
                "nn.conv2d",
                "nn.conv2d_transpose"
            ]
        )
        if isinstance(call.op, tir.op.Op):
            if str(call.op) in compute_intensive_ops:
                self.is_compute_intensive = True
        return super().visit_call(call)

    def is_compute_intensive_graph(self, subgraph) -> bool:
        self.visit(subgraph)
        return self.is_compute_intensive


def prune_no_mac_subgraphs(mod, compiler_name = "mera"):
    """Remove subgraphs that have no multiply-accumulates."""

    class SubgraphRemover(ExprMutator):
        def __init__(self, subgraphs_to_remove, mod):
            ExprMutator.__init__(self)
            self.subgraphs_to_remove = subgraphs_to_remove
            self.mod = mod

        def visit_call(self, call):
            if isinstance(call.op, GlobalVar):
                name = call.op.name_hint
                if name in self.subgraphs_to_remove:
                    # "Inline" the subgraph back into new main function
                    func = self.mod[name]
                    var_map = {}
                    for arg, param in zip(call.args, func.params):
                        var_map[param] = super().visit(arg)
                    new_body = bind(func.body, var_map)
                    return new_body
                if name != "main":
                    args = []
                    for arg in call.args:
                        args.append(super().visit(arg))
                    return call.op(*args)
            return super().visit_call(call)

    class InlineFunction(ExprMutator):
        def __init__(self):
            ExprMutator.__init__(self)

        def visit_call(self, call):
            if isinstance(call.op, Function):
                func = call.op
                var_map = {}
                for arg, param in zip(call.args, func.params):
                    var_map[param] = super().visit(arg)
                new_body = bind(func.body, var_map)
                return new_body
            return super().visit_call(call)

    subgraphs_to_remove = []
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or "Compiler" not in mod[name].attrs or mod[name].attrs["Compiler"] != compiler_name:
            continue
        if not IsComputeIntensiveGraph().is_compute_intensive_graph(mod[name].body):
            subgraphs_to_remove.append(name)
    # Create new pruned module
    new_mod = IRModule(mod.functions, mod.type_definitions)
    new_mod["main"] = SubgraphRemover(subgraphs_to_remove, mod).visit(mod["main"])
    new_mod["main"] = InlineFunction().visit(new_mod["main"])
    new_mod = transform.RemoveUnusedFunctions()(new_mod)
    return new_mod


def gather_captured_stats_pass(mod):
    """Generates statistics about captured nodes for MERA. """
    class CaptureStatsCollectorVisitor(ExprVisitor):
        def __init__(self):
            super().__init__()
            self.data = {}

        def __add(self, op):
            if op not in self.data:
                self.data[op] = 0
            self.data[op] = self.data[op] + 1

        def visit_call(self, call):
            if not isinstance(call.op, Function) and not isinstance(call.op, GlobalVar):
                self.__add(str(call.op))
            return super().visit_call(call)

        def run(self, mod):
            self.visit(mod)
            return self.data
    data = {}
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        codegen = mod[name].attrs["Compiler"] if mod[name].attrs and "Compiler" in mod[name].attrs else "TVM"
        if codegen not in data:
            data[codegen] = {}
        data[codegen][name] = CaptureStatsCollectorVisitor().run(mod[name])
    return data

def report_captured_stats(stats):
    total_captured = 0
    total_uncaptured = 0
    def __calc_captured(d):
        x = 0
        for sg_data in d.values():
            for op_count in sg_data.values():
                x += int(op_count)
        return x
    captured_regions = 0
    for codegen, cdgen_data in stats.items():
        if codegen == "TVM":
            total_uncaptured = __calc_captured(cdgen_data)
        else:
            captured_regions = len(cdgen_data.values())
            total_captured += __calc_captured(cdgen_data)
    total = total_captured + total_uncaptured
    logger.info(f'MERA model report: Captured {total_captured} operators and left {total_uncaptured} uncaptured '
        + f'({total_captured / total * 100:.1f}%) in {captured_regions} MERA regions.')


def _build_common(mod, params, target_tvm, fcompile, layout, output_dir, aux_config):
    assert get_global_func("relay.ext.mera")

    cfg = BuildConfig.current
    assert cfg.compiler_config["target"] is not None

    # TODO - Remove need to use API calls for set cfg
    ccfg_str, arch_cfg_str = _set_config(cfg)

    if aux_config["with_clip_pattern"]:
        pattern_table = get_pattern_table("mera_with_clip")
    elif aux_config["with_fc_pattern"]:
        pattern_table = get_pattern_table("mera_with_fc")
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

            desired_layout = {
                "qnn.conv2d": ["NHWC", "default"],
                "image.resize2d": ["NHWC"],
                "nn.max_pool2d": ["NHWC"]
            }
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
            # [!] 'mera_blocks' pattern table must run before 'mera' qtz table.
            transform.MergeComposite(get_pattern_table("mera_blocks")),
            transform.MergeComposite(pattern_table),
            transform.AnnotateTarget(["mera", "mera_blocks"]),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InlineExternClip(),
            transform.RemoveUnusedFunctions(),
        ]

        if aux_config["prequantize_input"]:
            pass_list.append(transform.RemoveInputQuantize())

        composite_partition = Sequential(pass_list)

        mod = composite_partition(mod)
        mod = prune_no_mac_subgraphs(mod)
        report_captured_stats(gather_captured_stats_pass(mod))

    input_layout = "NHWC"
    config = {
        "relay.ext.mera.options": {
            "input_layout": input_layout,
            "weight_layout": aux_config["weight_layout"],
            "mera_ccfg": ccfg_str,
            "mera_arch": arch_cfg_str
        }
    }

    with PassContext(opt_level=3, disabled_pass=["AlterOpLayout"], config=config):
        try:
            json, lib, all_params = _build(mod, target=target_tvm, params=params)
        except TVMError as ex:
            raise MeraError(f"ERROR found during compilation of MERA model:\n'{str(ex)}'")\
                .with_traceback(sys.exc_info()[2]) from ex

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

    if "with_fc_pattern" not in aux_config:
        aux_config["with_fc_pattern"] = False

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

def __run_passes_core(mod, params, target, host_arch, c_cfg, arch_cfg, aux_config, compiler_name):
    if host_arch == "x86":
        target_tvm = "llvm -mcpu=core-avx2"
    else:
        assert host_arch == "arm"
        target_tvm = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
    assert get_global_func(f"relay.ext.{compiler_name}")
    pattern_table = get_pattern_table(compiler_name)

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    with PassContext(opt_level=3):
        if "insert_downsample" in aux_config:
            new_in_height, new_in_width = aux_config["insert_downsample"]
            mod = transform.InsertResize(new_in_height, new_in_width)(mod)

        # convert layout to NHWC
        # if the original layout is NHWC, it has no effect.
        # NOTE: Even though we convert to OIHW for MERA, if there is a conv that is left uncaptured
        # TVM wont be able to run it with that weight layout.
        desired_layouts = {
            'nn.conv2d': ["NHWC", 'HWIO'],
            'nn.conv2d_transpose' : ['NHWC', 'HWIO'],
            'image.resize2d': ['NHWC'],
            'nn.max_pool2d': ['NHWC'],
            'nn.avg_pool2d': ['NHWC'],
            'nn.global_avg_pool2d': ['NHWC'],
        }
        seq = Sequential([transform.FoldExplicitPadding(),
                          transform.RemoveUnusedFunctions(),
                          transform.ConvertLayout(desired_layouts)])

        with PassContext(opt_level=3):
            mod = seq(mod)

        # Unfold BatchNorm and leave SimplifyInference to optimize it out
        seq = Sequential([
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.RemoveUselessPadding(),
            transform.FoldExplicitPadding(),
            transform.BackwardFoldScaleAxis(),
            transform.ForwardFoldScaleAxis(),
            transform.FoldConstant(),
            transform.DynamicToStatic(),
            transform.FoldMulAddToBN(),
            transform.FoldConstant(),
            transform.SimplifyInference(),
        ])
        with PassContext(opt_level=3):
            mod = seq(mod)

        # BYOC
        pass_list = [
            transform.MergeComposite(pattern_table),
            transform.AnnotateTarget(compiler_name),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InlineExternClip(),
            transform.RemoveUnusedFunctions(),
        ]

        if aux_config.get("prequantize_input", False):
            pass_list.append(transform.RemoveInputQuantize())

        composite_partition = Sequential(pass_list)

        mod = composite_partition(mod)
        mod = prune_no_mac_subgraphs(mod, compiler_name)
        report_captured_stats(gather_captured_stats_pass(mod))

    config = {
        "relay.ext.mera.options": {
            "mera_ccfg": c_cfg,
            "mera_arch": arch_cfg,
            "mera_target": target
        }
    }

    with PassContext(opt_level=3, config=config):
        try:
            json, lib, all_params = _build(mod, target=target_tvm, params=params)
        except TVMError as ex:
            raise MeraError(f"ERROR found during compilation of MERA model:\n'{str(ex)}'")\
                .with_traceback(sys.exc_info()[2]) from ex
    # remove constants that are embedded in Mera IR
    params = {}
    mera_globals = [x.name_hint for x in mod.get_global_vars()
        if mod[x.name_hint].attrs and mod[x.name_hint].attrs["Compiler"] == compiler_name]
    for k, v in all_params.items():
        if not np.any([str(k).startswith(mg) for mg in mera_globals]):
            params[k] = v
    return json, lib, params, mod


def build_fp32(mod, params, target, host_arch="x86", output_dir = None):
    """Build in fp32 precision and NHWC/IOHW layout"""
    if target not in ['Quantizer', 'Interpreter', 'InterpreterHwBf16']:
        raise ValueError(f'Unsupported target for fp32 MERA deployment: {target}')
    cfg = BuildConfig()
    cfg.compiler_config['target'] = target
    ccfg_str, arch_str = parse_config(cfg)
    json, lib, new_params, new_mod = __run_passes_core(mod, params, target, host_arch, ccfg_str, arch_str, {}, "mera_fp32")
    if output_dir:
        # Export data to output dir
        if host_arch == "x86":
            fcompile = cc.create_shared
        else:
            assert host_arch == "arm"
            fcompile = cc.cross_compiler("aarch64-linux-gnu-g++")
        lib_path = os.path.join(output_dir, "deploy.so")
        with open(os.path.join(output_dir, "deploy.json"), "w") as f:
            f.write(json)
        with open(os.path.join(output_dir, "deploy.params"), "wb") as f:
            f.write(save_param_dict(new_params))

        lib.export_library(lib_path, fcompile=fcompile)
    return json, lib, new_params, new_mod


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
