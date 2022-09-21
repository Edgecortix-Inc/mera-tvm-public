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
#ifndef TVM_RELAY_BACKEND_MERA_UTILS_H
#define TVM_RELAY_BACKEND_MERA_UTILS_H

#include <mera/mdna_ir.h>
#include "../../utils.h"

namespace tvm::relay::contrib {

template<typename T>
inline const T *AsChecked(const ObjectRef &ref) {
  const T *r_ptr = ref.as<T>();
  CHECK_NOTNULL(r_ptr);
  return r_ptr;
}

inline int GetInt(const tvm::PrimExpr &expr) {
  return expr.as<IntImmNode>()->value;
}

mera::ir::Padding AttrToMeraPadding(const Array<tvm::PrimExpr> &pad_attr);
mera::ir::Strides AttrToMeraStrides(const Array<tvm::PrimExpr> &stride_attr);
mera::ir::Dilations AttrToMeraDilations(const Array<tvm::PrimExpr> &dil_attr);

mera::ir::DataType ToMeraType(const tvm::DataType tvm_dtype);
mera::ir::DataType GetType(const Type& type);

mera::ir::Shape GetShape(const Type& type);
inline mera::ir::Shape GetShape(const CallNode *call) { return GetShape(call->checked_type()); }
mera::ir::Shape GetShapeNchw(const Type& type);
inline mera::ir::Shape GetShapeNchw(const CallNode *call) { return GetShapeNchw(call->checked_type()); }

} // namespace tvm::relay::contrib

#endif // TVM_RELAY_BACKEND_MERA_UTILS_H
