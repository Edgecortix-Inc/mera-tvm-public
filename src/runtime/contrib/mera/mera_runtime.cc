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
 * \file mera_runtime.cc
 */

#include "mera_runtime.h"

#include <mera/mdna_execute.h>
#include <mera/mdna_simulate.h>
#include <mera/mdna_version.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

MeraRuntime::MeraRuntime(std::vector<uint8_t> code, bool interpreter, const std::string& func_name)
    : code_(code), interpreter_(interpreter), func_name_(func_name) {
}

void MeraRuntime::Init(mera::execute::DeviceRunTarget device_run_target) {
  mera_exec_ = mera::execute::CreateExecutor(code_, device_run_target);
}

void MeraRuntime::Invoke() {}

std::string MeraRuntime::GetSource(const std::string& format) {
  return std::string();
}

void MeraRuntime::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(static_cast<uint64_t>(code_.size()));
  stream->Write(static_cast<bool>(interpreter_));
  stream->Write(static_cast<std::string>(func_name_));
  stream->WriteArray(this->code_.data(), code_.size());
}

PackedFunc MeraRuntime::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int in_idx = args[0];
      CHECK_GE(in_idx, 0);
    });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
    });
  } else if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->Invoke();
    });
  } else if (name == "get_runtime_metrics") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = all_metrics_.AsString(mera::execute::ExecutorMetrics::MetricsType::RUNTIME); });
  } else if (name == "get_power_metrics") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = all_metrics_.AsString(mera::execute::ExecutorMetrics::MetricsType::POWER); });
  } else if (name == func_name_) {
    return PackedFunc([sptr_to_self, name, this](TVMArgs args, TVMRetValue* rv) {
      std::vector<void*> argument_data;
      for (int i = 0; i < args.size(); i++) {
        DLTensor* arg = static_cast<DLTensor*>(args[i]);
        CHECK(arg);
        argument_data.push_back(arg->data);
      }
      all_metrics_ = mera::execute::Execute(mera_exec_.get(), name, argument_data);
    });
  } else if (name == "mera_get_interpreter_buffer") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 1);
      CHECK(interpreter_) << "Function only available for Interpreters";
      const auto *int_ptr = dynamic_cast<const mera::interpreter::Interpreter_*>(mera_exec_.get());
      CHECK_NOTNULL(int_ptr);
      GetInterpreterBufferImpl(rv, int_ptr, args[0].operator std::string());
    });
  } else if (name == "mera_get_interpreter_node_list") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK(interpreter_) << "Function only available for Interpreters";
      const auto *int_ptr = dynamic_cast<const mera::interpreter::Interpreter_*>(mera_exec_.get());
      CHECK_NOTNULL(int_ptr);
      GetInterpreterNodeListImpl(rv, int_ptr);
    });
  } else if (name == "mera_runtime_init_device") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 1);
      int rt_target = static_cast<int>(args[0]);
      auto runtime_target = static_cast<mera::execute::DeviceRunTarget>(rt_target);
      //LOG(INFO) << "Init with " << runtime_target;
      this->Init(runtime_target);
    });
  } else {
    return PackedFunc();
  }
}

Module MeraRuntimeCreateEmpty(std::vector<uint8_t> code, bool interpreter, const std::string& func_name) {
  auto exec = make_object<MeraRuntime>(code, interpreter, func_name);
  return Module(exec);
}

Module MeraRuntimeCreate(std::vector<uint8_t> code, bool interpreter, const std::string& func_name) {
  auto exec = make_object<MeraRuntime>(code, interpreter, func_name);
  exec->Init(mera::execute::DeviceRunTarget::NONE);
  return Module(exec);
}

Module LoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  uint64_t size;
  bool use_interpreter;
  stream->Read(&size);
  stream->Read(&use_interpreter);
  std::string func_name;
  stream->Read(&func_name);
  std::vector<uint8_t> code;
  code.resize(size);
  stream->ReadArray(code.data(), code.size());
  return MeraRuntimeCreateEmpty(code, use_interpreter, func_name);
}

void GetInterpreterBufferImpl(TVMRetValue *rv, const mera::interpreter::Interpreter_ *impl, const std::string &op_id) {
  auto buf_data = impl->GetInterpreterBuffer(op_id);

  if (buf_data.has_value()) {
    const auto shape = buf_data->shape;

    ShapeTuple tvm_shape(shape.begin(), shape.end());
    DataType tvm_type;
    switch (buf_data->type) {
      case mera::ir::DataType::Float32: tvm_type = DataType::Float(32); break;
      case mera::ir::DataType::Int32: tvm_type = DataType::Int(32); break;
      case mera::ir::DataType::Int8: tvm_type = DataType::Int(8); break;
      case mera::ir::DataType::UInt8: tvm_type = DataType::UInt(8); break;
      default: LOG(FATAL) << "Unknown data type";
    }

    auto data = NDArray::Empty(tvm_shape, tvm_type, {DLDeviceType::kDLCPU, 0});
    data.CopyFromBytes(buf_data->data, buf_data->size * tvm_type.bytes());
    *rv = data;
  }
}

void GetInterpreterNodeListImpl(TVMRetValue *rv, const mera::interpreter::Interpreter_ *impl) {
  std::stringstream ss;
  const auto node_list = impl->GetInterpreterNodeList();
  ss << '[';
  for (size_t i = 0; i < node_list.size(); ++i) {
    const auto &n = node_list[i];
    ss << "[\"" << n.id << "\",\"" << n.op_type << "\"]";
    if (i != node_list.size() - 1) {
      ss << ',';
    }
  }
  ss << ']';
  *rv = ss.str();
}

template<typename B>
std::unique_ptr<mera::blocks::MeraBlock> LoadMeraBlock(const std::vector<uint8_t> &params) {
  std::unique_ptr<mera::blocks::MeraBlock> block{std::make_unique<B>()};
  block->LoadParams(params);
  return std::move(block);
}

std::unique_ptr<mera::blocks::MeraBlock> GetBlockImpl(const std::string &block_id, const std::vector<uint8_t> &params) {
  if (block_id == mera::blocks::Yolov5Post::GetBlockId()) {
    return std::move(LoadMeraBlock<mera::blocks::Yolov5Post>(params));
  } else if (block_id == mera::blocks::Yolov5i8Post::GetBlockId()) {
    return std::move(LoadMeraBlock<mera::blocks::Yolov5i8Post>(params));
  } else {
    LOG(FATAL) << "Unknown MeraBlock ID " << block_id;
    return nullptr;
  }
}

MeraBlocksRuntime::MeraBlocksRuntime(const std::string &func_name, const std::string &block_id,
  const std::vector<uint8_t> &compiled_code): func_name_(func_name), block_id_(block_id), compiled_code_(compiled_code),
  impl_ptr_(std::move(GetBlockImpl(block_id, compiled_code))) {}

Module MeraBlocksCreate(const std::string &func_name, const std::string &block_id,
  const std::vector<uint8_t> &compiled_code) {
  return Module(make_object<MeraBlocksRuntime>(func_name, block_id, compiled_code));
}

void MeraBlocksRuntime::SaveToBinary(dmlc::Stream *stream) {
  stream->Write(func_name_);
  stream->Write(block_id_);
  stream->Write(static_cast<uint64_t>(compiled_code_.size()));
  stream->WriteArray(compiled_code_.data(), compiled_code_.size());
}

Module LoadFromBinaryBlocks(void *strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string func_name;
  stream->Read(&func_name);
  std::string block_id;
  stream->Read(&block_id);
  uint64_t size;
  stream->Read(&size);
  std::vector<uint8_t> code(size);
  stream->ReadArray(code.data(), size);
  return MeraBlocksCreate(func_name, block_id, code);
}

PackedFunc MeraBlocksRuntime::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == func_name_) {
    // Return func for executing the block
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_NOTNULL(impl_ptr_);
        std::vector<void*> argument_data;
        for (int i = 0; i < args.size(); i++) {
          DLTensor* arg = static_cast<DLTensor*>(args[i]);
          CHECK(arg);
          argument_data.push_back(arg->data);
        }
        impl_ptr_->Evaluate(argument_data);
    });
  }
  return PackedFunc();
}


TVM_REGISTER_GLOBAL("runtime.module.loadbinary_MeraRuntime").set_body_typed(LoadFromBinary);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_MeraBlocksRuntime").set_body_typed(LoadFromBinaryBlocks);

TVM_REGISTER_GLOBAL("runtime.module.get_version").set_body_typed([]() { return mera::GetMeradnaVersionStr(); });

TVM_REGISTER_GLOBAL("runtime.module.mera_module_create_empty")
    .set_body_typed([](DLTensor* code, bool interpreter, const std::string& func_name) {
      auto* begin = reinterpret_cast<const uint8_t*>(code->data);
      return MeraRuntimeCreateEmpty(std::vector<uint8_t>(begin, begin + code->shape[0]), interpreter,
                             func_name);
    });

TVM_REGISTER_GLOBAL("runtime.module.loadfile_mera").set_body_typed([](DLTensor* code) {
  return MeraRuntimeCreate(std::vector<uint8_t>(), true, "");
});

TVM_REGISTER_GLOBAL("runtime.module.mera_blocks_module_create")
  .set_body_typed([](const std::string &func_name, const std::string &block_id, DLTensor* code) {
    auto* begin = reinterpret_cast<const uint8_t*>(code->data);
    return MeraBlocksCreate(func_name, block_id, std::vector<uint8_t>(begin, begin + code->shape[0]));
  });

}  // namespace runtime
}  // namespace tvm
