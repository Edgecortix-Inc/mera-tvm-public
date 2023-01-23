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
import sys
import os
import random

import numpy as np
import torch
import yolov3

from torch import nn
from torch.quantization import fuse_modules, QuantWrapper
from torch.quantization import QuantStub, DeQuantStub

from tvm import relay
from tvm.relay import mera

# clone pytorch_quantization at the same dir as private-tvm
sys.path.append("../../../../pytorch_quantization/tvm_qnn_evaluation")
sys.path.append("../../../../pytorch_quantization/models")
sys.path.append("../frontend/pytorch")
sys.path.append(".")

from test_ec_compiler import load_or_quantize
from test_ec_compiler import nchw_to_nhwc


def test_execute_end_to_end(file_name, model, dummy_calib=False, with_clip_pattern=False):
    inp = torch.rand((1, 3, 288, 576))
    script_module = load_or_quantize(file_name, model, inp, dummy_calib)

    with mera.build_config(target="InterpreterHw"):
        int_result = mera.test_util.run_mera_backend(script_module, inp, layout="NCHW")

    int_result_nhwc = [nchw_to_nhwc(int_result[0]), nchw_to_nhwc(int_result[1])]
    with mera.build_config(target="Simulator"):
        mera.test_util.run_mera_backend(
            script_module,
            nchw_to_nhwc(inp),
            int_result_nhwc,
            layout="NHWC",
        )


def test_yolov3():
    model_func = lambda: yolov3.hub.load('yolov3-tiny', quantizable=True, fuse_model=False)
    test_execute_end_to_end("tiny_yolov3.pt", model_func, dummy_calib=True)


if __name__ == "__main__":
    torch.manual_seed(123)
    test_yolov3()
