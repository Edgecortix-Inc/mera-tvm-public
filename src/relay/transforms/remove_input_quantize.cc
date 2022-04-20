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
 * \file remove_input_quantize.cc
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

using VarMap = std::unordered_map<Expr, Var, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

class FillVarReplacementMap : public ExprVisitor {
 public:
  FillVarReplacementMap() : qnn_quantize_op_(Op::Get("qnn.quantize")) {}

  void VisitExpr_(const CallNode* n) {
    if (n->op == qnn_quantize_op_) {
      auto quantized_input = n->args[0];
      if (quantized_input->IsInstance<VarNode>()) {
        auto var = Downcast<Var>(quantized_input);
        auto tensor_type = Downcast<TensorType>(var->type_annotation);
        const auto attrs = n->attrs.as<qnn::QuantizeAttrs>();
        const auto output_dtype = attrs->out_dtype;
        if (tensor_type.defined()) {
          auto new_tensor_type = TensorType(tensor_type->shape, output_dtype);
          var_replacement_[quantized_input] = Var(var->name_hint(), new_tensor_type);
        }
      }
    }
    ExprVisitor::VisitExpr_(n);
  }

  VarMap GetResult() const { return var_replacement_; }

 private:
  VarMap var_replacement_;
  const Op& qnn_quantize_op_;
};

class RemoveInputQuantizeTransformer : public ExprMutator {
 public:
  RemoveInputQuantizeTransformer(VarMap var_replacement)
      : var_replacement_(var_replacement), qnn_quantize_op_(Op::Get("qnn.quantize")) {}

  Expr VisitExpr_(const FunctionNode* n) {
    tvm::Array<Var> new_params;
    for (auto p : n->params) {
      auto new_input_var = var_replacement_.find(p);
      CHECK(new_input_var != var_replacement_.end());
      new_params.push_back(new_input_var->second);
    }
    auto new_body = ExprMutator::VisitExpr(n->body);
    return Function(new_params, new_body, n->ret_type, n->type_params);
  }

  Expr VisitExpr_(const CallNode* n) {
    if (n->op == qnn_quantize_op_) {
      auto quantized_input = n->args[0];
      auto new_input_var = var_replacement_.find(quantized_input);
      if (new_input_var != var_replacement_.end()) {
        return new_input_var->second;
      }
    }
    return ExprMutator::VisitExpr_(n);
  }

 private:
  VarMap var_replacement_;
  const Op& qnn_quantize_op_;
};

Expr RemoveInputQuantize(const Expr& e) {
  FillVarReplacementMap filler;
  filler.VisitExpr(e);
  return RemoveInputQuantizeTransformer(filler.GetResult()).Mutate(e);
}

namespace transform {

Pass RemoveInputQuantize() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto main_func = m->Lookup("main");
        if (main_func == f) {
          return Downcast<Function>(RemoveInputQuantize(f));
        }
        return f;
      };
  return CreateFunctionPass(pass_func, 0, "RemoveInputQuantize", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.RemoveInputQuantize").set_body_typed(RemoveInputQuantize);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
