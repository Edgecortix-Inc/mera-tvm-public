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
 * \brief Mera runtime containing only tvm PackedFunc.
 * \file mera_runtime.h
 */
#ifndef TVM_RUNTIME_CONTRIB_EC_EC_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_EC_EC_RUNTIME_H_

#include <dlpack/dlpack.h>
#include <mera/mdna_execute.h>
#include <mera/mdna_interpreter.h>
#include <mera/mdna_blocks.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Mera runtime.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime PackedFunc API.
 */
class MeraRuntime : public ModuleNode {
 public:
  explicit MeraRuntime(std::vector<uint8_t> code, bool interpreter, const std::string& func_name);

  std::string GetSource(const std::string& format = "");

  void SaveToBinary(dmlc::Stream* stream) final;

  /*!
   * \brief Get member function to front-end.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const { return "MeraRuntime"; }

  /*!
   * \brief Invoke the internal mera interpreter and run the whole model in
   * dependency order.
   */
  void Invoke();

  /*!
   * \brief Initialize the mera runtime with context.
   */
  void Init(mera::execute::DeviceRunTarget device_run_target);

 private:
  std::unique_ptr<mera::execute::Executor> mera_exec_;
  std::vector<uint8_t> code_;
  bool interpreter_;
  std::string func_name_;
  mera::execute::ExecutorMetrics all_metrics_;
};

void GetInterpreterBufferImpl(TVMRetValue *rv, const mera::interpreter::Interpreter_ *impl, const std::string &op_id);

void GetInterpreterNodeListImpl(TVMRetValue *rv, const mera::interpreter::Interpreter_ *impl);

/*!
 * \brief MERA Blocks runtime.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime PackedFunc API.
 */
class MeraBlocksRuntime : public ModuleNode {
 public:
  explicit MeraBlocksRuntime(const std::string &func_name, const std::string &block_id, const std::vector<uint8_t> &compiled_code);

  std::string GetSource(const std::string&) { return ""; }

  void SaveToBinary(dmlc::Stream *stream) final;

  virtual PackedFunc GetFunction(const std::string &name, const ObjectPtr<Object> &sptr_to_self);

  const char* type_key() const { return "MeraBlocksRuntime"; }
 private:
  std::string func_name_;
  std::string block_id_;
  std::vector<uint8_t> compiled_code_;
  std::unique_ptr<mera::blocks::MeraBlock> impl_ptr_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_EC_EC_RUNTIME_H_
