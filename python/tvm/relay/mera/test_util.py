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
import time
import numpy as np

from ..backend import compile_engine
from ... import testing
from .. import transform
from .build_module import build as ec_build
from .build_module import load_runtime_module
from ..build_module import build as _build
from ..frontend import from_pytorch
from ...contrib import graph_runtime
from ... import runtime


def verify_output(ref_result, ec_result, tol=1e-5, quantized=True):
    if quantized:
        max_abs_diff = np.max(np.abs(ec_result - ref_result))
        mean_abs_diff = np.mean(np.abs(ec_result - ref_result))
        num_identical = np.sum(ec_result == ref_result)
        print("max abs diff:", max_abs_diff)
        print("mean abs_diff:", mean_abs_diff)
        print("correct ratio:", num_identical / np.prod(ec_result.shape))

        if len(ref_result[0].shape) == 1:
            # imagenet output
            print("Ref top5 label:", np.argsort(ref_result[0])[::-1][:5])
            print("EC top5 label:", np.argsort(ec_result[0])[::-1][:5])
    else:
        testing.assert_allclose(ec_result, ref_result, rtol=tol, atol=tol)


def run_and_verify(rt_mod, input_name, inp_np, ref_results=None, tol=1e-5, quantized=True):
    rt_mod.set_input(input_name, inp_np)

    outputs = []
    t1 = time.time()
    rt_mod.run()
    for i in range(rt_mod.get_num_outputs()):
        outputs.append(rt_mod.get_output(i).asnumpy())
    t2 = time.time()

    print("Elapsed %f seconds" % (t2 - t1))

    if ref_results is not None:
        if not isinstance(ref_results, list):
            ref_results = [ref_results]

        for ref_result, ec_result in zip(ref_results, outputs):
            verify_output(ref_result, ec_result, tol=tol, quantized=quantized)

    if len(outputs) == 1:
        return outputs[0]

    return outputs


def run_mera_backend(
    script_module, inp_np, ref_results=None, layout="NCHW", output_dir="_out", aux_config={}
):
    compile_engine.get().clear()
    input_name = "input0"
    input_shapes = [(input_name, inp_np.shape)]
    mod, params = from_pytorch(script_module, input_shapes, layout=layout)

    json, params, lib_path = ec_build(
        mod, params, host_arch="x86", output_dir=output_dir, layout=layout, aux_config=aux_config
    )
    rt_mod = load_runtime_module(json, params, lib_path)

    return run_and_verify(rt_mod, input_name, inp_np, ref_results=ref_results)


def run_tvm(script_module, inp_np):
    compile_engine.get().clear()
    input_name = "input0"
    input_shapes = [(input_name, inp_np.shape)]
    mod, params = from_pytorch(script_module, input_shapes)

    target = "llvm -mcpu=core-avx2"

    with transform.build_config(opt_level=3):
        json, lib, params = _build(mod, target=target, params=params)

    rt_mod = graph_runtime.create(json, lib, runtime.cpu())
    rt_mod.set_input(**params)

    rt_mod.set_input(input_name, inp_np)
    rt_mod.run()

    return rt_mod.get_output(0).asnumpy()
