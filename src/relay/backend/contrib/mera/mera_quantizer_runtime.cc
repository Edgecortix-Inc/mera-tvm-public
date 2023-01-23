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

/*!
 * \file mera_quantizer_runtime.cc
 */

#include "mera_quantizer_runtime.h"

#include <sstream>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include "../../../../runtime/contrib/mera/mera_runtime.h"

namespace tvm {
namespace runtime {

PackedFunc MeraQuantizerRuntime::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == func_name_) {
    return PackedFunc([sptr_to_self, name, this](TVMArgs args, TVMRetValue* rv) {
      std::vector<void*> argument_data;
      for (int i = 0; i < args.size(); i++) {
        DLTensor* arg = static_cast<DLTensor*>(args[i]);
        CHECK(arg);
        argument_data.push_back(arg->data);
      }
      mera_quant_->RunCalibrationImage(argument_data);
    });
  } else if (name == func_name_ + "_transform") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      auto transformed_ir = mera_quant_->QuantizeTransform();
      auto data = NDArray::Empty({static_cast<int64_t>(transformed_ir.size())}, DataType::UInt(8), {DLDeviceType::kDLCPU, 0});
      data.CopyFromBytes(transformed_ir.data(), transformed_ir.size());
      *rv = data;
    });
  } else if (name == "mera_quantizer_reset") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      mera_quant_->Reset();
    });
  } else if (name == "mera_calculate_qparams") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = mera_quant_->CalculateQParams();
    });
  } else if (name == "mera_get_interpreter_buffer") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 1);
      GetInterpreterBufferImpl(rv, mera_quant_.get(), args[0].operator std::string());
    });
  } else if (name == "mera_get_interpreter_node_list") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      GetInterpreterNodeListImpl(rv, mera_quant_.get());
    });
  } else {
    return PackedFunc();
  }
}

Module MeraQuantizerCreate(const std::vector<uint8_t> &code, const std::string& func_name) {
  auto exec = make_object<MeraQuantizerRuntime>(code, func_name);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("runtime.module.mera_quantizer_create")
    .set_body_typed([](DLTensor* code, const std::string& func_name) {
      auto* begin = reinterpret_cast<const uint8_t*>(code->data);
      return MeraQuantizerCreate(std::vector<uint8_t>(begin, begin + code->shape[0]), func_name);
    });

}  // namespace runtime
}  // namespace tvm
