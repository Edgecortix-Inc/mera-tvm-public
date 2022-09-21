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
 * \file remove_useless_padding.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/attrs/nn.h>

namespace tvm {
namespace relay {

bool IsUselessPadding(const PadAttrs* param) {
  for (size_t i = 0; i < param->pad_width.size(); ++i) {
    for (size_t j = 0; j < param->pad_width[0].size(); ++j) {
      if (int(param->pad_width[i][j]) > 0) {
        return false;
      }
    }
  }
  return true;
}

class UselessPaddingRemover : public MixedModeMutator {
 public:
  UselessPaddingRemover() {}

  Expr Rewrite_(const CallNode* n, const Expr& new_n) {
    if (n->op == Op::Get("nn.pad")) {
      const PadAttrs* param = n->attrs.as<PadAttrs>();
      if (IsUselessPadding(param)) {
        return new_n.as<CallNode>()->args[0];
      }
    }
    return new_n;
  }
};

Expr RemoveUselessPadding(const Expr& e) { return UselessPaddingRemover().Mutate(e); }

namespace transform {

Pass RemoveUselessPadding() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(RemoveUselessPadding(f));
      };
  return CreateFunctionPass(pass_func, 0, "RemoveUselessPadding", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.RemoveUselessPadding").set_body_typed(RemoveUselessPadding);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
