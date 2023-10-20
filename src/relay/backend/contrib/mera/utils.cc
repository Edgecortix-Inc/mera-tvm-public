/*
 * Copyright 2022 EdgeCortix Inc
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "utils.h"

namespace tvm::relay::contrib {

mera::ir::Padding AttrToMeraPadding(const Array<tvm::PrimExpr> &pad_attr) {
  const int p_top = GetInt(pad_attr[0]);
  const int p_left = GetInt(pad_attr[1]);
  const int p_bottom = GetInt(pad_attr[2]);
  const int p_right = GetInt(pad_attr[3]);
  return mera::ir::Padding{p_top, p_bottom, p_left, p_right};
}

mera::ir::Strides AttrToMeraStrides(const Array<tvm::PrimExpr> &stride_attr) {
  return mera::ir::Strides{GetInt(stride_attr[0]), GetInt(stride_attr[1])};
}

mera::ir::Dilations AttrToMeraDilations(const Array<tvm::PrimExpr> &dil_attr) {
  return mera::ir::Dilations{GetInt(dil_attr[0]), GetInt(dil_attr[1])};
}

mera::ir::DataType ToMeraType(const tvm::DataType tvm_dtype) {
  // TODO
  if (tvm_dtype.is_float()) {
    return mera::ir::DataType::Float32;
  } else if (tvm_dtype.is_int() && tvm_dtype.bits() == 32) {
    return mera::ir::DataType::Int32;
  } else if (tvm_dtype.is_int() && tvm_dtype.bits() == 8) {
    return mera::ir::DataType::Int8;
  } else if (tvm_dtype.is_uint() && tvm_dtype.bits() == 8) {
    return mera::ir::DataType::UInt8;
  }

  LOG(FATAL) << "Cannot convert dtype " << tvm_dtype;
  return mera::ir::DataType::Float32;
}

mera::ir::DataType GetType(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK_NOTNULL(ttype);
  return ToMeraType(ttype->dtype);
}

mera::ir::Shape GetShape(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK_NOTNULL(ttype);
  std::vector<int> shape;
  int s = 1;
  for (unsigned i = 0; i < ttype->shape.size(); ++i) {
    auto* v = ttype->shape[i].as<IntImmNode>();
    CHECK_NOTNULL(v);
    shape.push_back(v->value);
    s *= v->value;
  }
  mera::ir::Layout layout;
  // Default layout allocations
  switch (shape.size()) {
    case 4: layout = mera::ir::layout::NCHW; break;
    case 3: layout = mera::ir::layout::NHW; break;
    case 2: layout = mera::ir::layout::HW; break;
    case 1: layout = mera::ir::layout::W; break;
    case 0: layout = mera::ir::layout::x; break;
    default: LOG(FATAL) << "Unsupported rank " << shape.size();
  }
  // Don't allow rank 0 shapes
  if (shape.size() == 0) {
    shape.push_back(1);
  }
  return mera::ir::Shape{shape, layout};
}

mera::ir::Shape ExpandTo4D(const mera::ir::Shape &shape) {
  if (shape.rank == 4) {
    return shape;
  }
  CHECK_LT(shape.rank, 4) << "Shapes with dim > 4 not allowed";
  mera::ir::Shape ret = shape;
  if (shape.rank == 3) {
    // XHW -> XHWX
    ret.rank = 4;
    ret.shape[3] = 1;
    ret.layout = mera::ir::layout::NHWC;
  } else {
    CHECK(false) << "Unhandled ExpandTo4d rank: " << shape.rank;
  }
  return ret;
}

mera::ir::Shape GetShapeNchw(const Type& type) {
  // All shapes coming from TVM are in NHWC, so convert and reorder
  auto shape_nhwc = GetShape(type);
  CHECK_EQ(shape_nhwc.rank, 4) << "Only 4D shapes allowed";
  return mera::ir::Shape{
    {shape_nhwc.shape[0], shape_nhwc.shape[3], shape_nhwc.shape[1], shape_nhwc.shape[2]},
    mera::ir::layout::NCHW};
}

} // namespace tvm::relay::contrib
