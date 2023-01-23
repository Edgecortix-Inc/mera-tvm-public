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
 * \brief MERA quantizer runtime containing only tvm PackedFunc.
 * \file mera_quantizer_runtime.h
 */
#ifndef TVM_RUNTIME_CONTRIB_MERA_QUANTIZER_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_MERA_QUANTIZER_RUNTIME_H_

#include <dlpack/dlpack.h>
#include <mera/mdna_quantize.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief MERA quantizer runtime.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime PackedFunc API.
 */
class MeraQuantizerRuntime : public ModuleNode {
 public:
  explicit MeraQuantizerRuntime(const std::vector<uint8_t> &code, const std::string& func_name):
    mera_quant_(mera::quantizer::CreateQuantizer(code)), code_(code), func_name_(func_name) {}
  virtual ~MeraQuantizerRuntime() {}

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
  virtual const char* type_key() const { return "MeraQuantizerRuntime"; }

  std::string GetFuncName() const { return func_name_; }
 private:
  std::unique_ptr<mera::quantizer::Quantizer> mera_quant_;
  const std::vector<uint8_t> code_;
  std::string func_name_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MERA_QUANTIZER_RUNTIME_H_
