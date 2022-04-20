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
from .common import infer_shape as _infer_shape


def _same_elements(lst):
    return all(map(lambda dim: dim == lst[0], lst))


def is_channel_concat(data):
    # Decide if this is concat along channel axis
    # TODO: Needs a more principled approach
    shapes = [_infer_shape(inp) for inp in data]
    if not all(map(lambda shape: len(shape) == 4, shapes)):
        return False

    h_dims = [shape[1] for shape in shapes]
    w_dims = [shape[2] for shape in shapes]
    return _same_elements(h_dims) and _same_elements(w_dims)


def map_axis(axis, layout):
    if layout == "NHWC":
        mapping = [0, 3, 1, 2]
        return mapping[axis]
    else:
        assert layout == "NCHW"
        return axis
