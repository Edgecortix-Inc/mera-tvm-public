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
from ... import IRModule
from ...ir.transform import PassContext
from ..function import Function
from ..expr import Constant, Tuple
import tvm.relay.op._make

def mera_quantize_result(mera_data, out_type, *args):
    input = Tuple(*args)
    return tvm.relay.op._make.mera_quantized_result(input, Constant(mera_data), out_type)

def pass_rename_sg_attr(mod, attr_name, from_val, to_val):
    """Alter the module by changing all subgraph attributes ['attr_name'] from 'from_val' to 'to_val'."""
    new_mod = IRModule(mod.functions, mod.type_definitions)
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if mod[name].attrs and attr_name in mod[name].attrs and mod[name].attrs[attr_name] == from_val:
            new_mod[name] = mod[name].with_attr(attr_name, to_val)
    return new_mod

def pass_internal_quantize(mod, qtzer_mod, qtz_compiler_name):
    """Replace all MERA functions by their internal quantized representation."""
    new_mod = IRModule(mod.functions, mod.type_definitions)
    tvm_qtz_params = {}
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if mod[name].attrs and mod[name].attrs["Compiler"] == qtz_compiler_name:
            fn = mod[name]
            qtzed_bin_ir = qtzer_mod["mera_quantizer_transform"](name)
            # There will only be 1 constant for this function
            tvm_qtz_params[name + "_const_0"] = qtzed_bin_ir
            # Collapse function into MERA IR representation
            fn_body = mera_quantize_result(qtzed_bin_ir, fn.ret_type, fn.params)
            new_mod[name] = Function(fn.params, fn_body, fn.ret_type, fn.type_params, fn.attrs)
    return new_mod, tvm_qtz_params


def qtz_transform(mod_f32, qtzer_mod):
    _QTZ_COMPILER_NAME = "mera_qtz"
    with PassContext(opt_level=3):
        # Make all MERA functions have the mera_qtz compiler
        mod = pass_rename_sg_attr(mod_f32, "Compiler", "mera_fp32", _QTZ_COMPILER_NAME)
        # Group MERA functions with the transformed quantized result
        mod, mera_qtzed_ir = pass_internal_quantize(mod, qtzer_mod, _QTZ_COMPILER_NAME)
    return mod, mera_qtzed_ir