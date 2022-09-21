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
#ifndef TVM_RELAY_BACKEND_MERA_COMPILER_H
#define TVM_RELAY_BACKEND_MERA_COMPILER_H

#include <mera/mdna_ir.h>
#include <mera/mdna_ir_io.h>
#include <tvm/relay/expr_functor.h>
#include <vector>
#include <functional>
#include <unordered_map>

#include "utils.h"
#include "../../utils.h"

namespace tvm::relay::contrib {

struct MeraCompilerConfigNode : public tvm::AttrsNode<MeraCompilerConfigNode> {
  String input_layout;
  String weight_layout;
  String mera_ccfg;
  String mera_arch;

  TVM_DECLARE_ATTRS(MeraCompilerConfigNode, "ext.attrs.MeraCompilerConfigNode") {
    TVM_ATTR_FIELD(input_layout).set_default("NHWC");
    TVM_ATTR_FIELD(weight_layout).set_default("OIHW");
    TVM_ATTR_FIELD(mera_ccfg).set_default("");
    TVM_ATTR_FIELD(mera_arch).set_default("");
  }
};

class MeraCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(MeraCompilerConfig, Attrs, MeraCompilerConfigNode);
};

MeraCompilerConfig GetMeraCompilerConfig();

using TensorVec_t = std::vector<mera::ir::Tensor>;
using Scope_t = std::map<std::string, mera::ir::Tensor>;

class IRContext {
  backend::MemoizedExprTranslator<TensorVec_t> &visitor;
  const CallNode *root_call;

public:
  IRContext(backend::MemoizedExprTranslator<TensorVec_t> &visitor, const CallNode *root_call):
    visitor(visitor), root_call(root_call) {}

  /**
   * @brief Utiliy class for IR traversals from the root of a composite.
   */
  class IRTraverse {
    const CallNode *curr_ir_pos;
    const IRContext &owner;
  public:
    IRTraverse(const CallNode *pos, const IRContext &owner): curr_ir_pos(pos), owner(owner) {}

    /**
     * @brief Move up the IR Call stack through argument 'index' of current position.
     */
    IRTraverse Get(unsigned index);

    /**
     * @brief Traverses the IR upstream via 'index' if the current IR position is of operation
     * 'opt_op_name', otherwise return current position.
     */
    IRTraverse MoveIf(const std::string &opt_op_name, unsigned index = 0);

    inline IRTraverse operator[](unsigned index) { return Get(index); }

    inline const CallNode *GetCall() const { return curr_ir_pos; }

    bool IsOp(const std::string &op_name) const;

    /**
     * @brief Generates the MERA constant node attached to argument 'arg_index' of the current IR position.
     * TVM constants are embedded inside the composite call so they never are inputs to the composite.
     */
    mera::ir::Tensor CompileConstant(unsigned arg_index) const;
  };

  /**
   * @brief Starts an IR traversal through the root call of this composite.
   */
  inline IRTraverse Traverse() const { return IRTraverse(root_call, *this); }

  inline const CallNode *GetRootCall() const { return root_call; }
};

class MeraCompilerBase {
  using CodeGenFunc_t = std::function<TensorVec_t(const TensorVec_t &, const IRContext &)>;
  friend class MeraCompilerVisitor;

protected:
  std::string ext_func_id_;
  mera::ir::Graph& graph_;
  std::map<std::string, CodeGenFunc_t> codegen_funcs_;

  MeraCompilerBase(const std::string& id, mera::ir::Module& module,
    const std::map<std::string, CodeGenFunc_t> &&funcs):
    ext_func_id_(id), graph_(module.AddFunction(id)), codegen_funcs_(funcs) {}

public:
  void Compile(const tvm::relay::Function &func);
};

/**
 * @brief MERA IR codegen spetialization for float32.
 */
struct MeraFp32Compiler : public MeraCompilerBase {
  constexpr static mera::ir::DataType kType = mera::ir::DataType::Float32;

  explicit MeraFp32Compiler(const std::string& id, mera::ir::Module& module);
};

} // namespace tvm::relay::contrib

#endif // MERA_RELAY__BACKEND_MERA_COMPILER_H
