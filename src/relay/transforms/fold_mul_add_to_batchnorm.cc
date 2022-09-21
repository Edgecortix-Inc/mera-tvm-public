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
 * \file fold_mul_add_to_batchnorm.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

Expr _SimplifyBNMulAdd(const CallNode* batch_norm_node, Expr scale, Expr shift) {
  Expr gamma = batch_norm_node->args[1];
  Expr beta = batch_norm_node->args[2];
  Expr moving_mean = batch_norm_node->args[3];
  Expr moving_var = batch_norm_node->args[4];
  Expr reshaped_scale = Call(Op::Get("reshape_like"), {scale, gamma}, Attrs(make_object<ReshapeLikeAttrs>()));
  Expr reshaped_shift = Call(Op::Get("reshape_like"), {shift, gamma}, Attrs(make_object<ReshapeLikeAttrs>()));
  const auto attrs = batch_norm_node->attrs.as<BatchNormAttrs>();
  Expr new_gamma = reshaped_scale;
  if (attrs->scale) {
    new_gamma = Multiply(gamma, new_gamma);
  }
  Expr new_beta = reshaped_shift;
  if (attrs->center) {
    new_beta = Add(Multiply(beta, reshaped_scale), new_beta);
  }
  auto new_attrs = make_object<BatchNormAttrs>();
  new_attrs->axis = attrs->axis;
  new_attrs->epsilon = attrs->epsilon;
  new_attrs->scale = true;
  new_attrs->center = true;
  Expr new_batch_norm = Call(batch_norm_node->op,
    {batch_norm_node->args[0], new_gamma, new_beta, moving_mean, moving_var},
    Attrs(new_attrs), batch_norm_node->type_args, batch_norm_node->span);
  return TupleGetItem(new_batch_norm, 0);
}

class SimplifyBNMulAdd : public MixedModeMutator {
 public:
  SimplifyBNMulAdd()
      : batch_norm_op_(Op::Get("nn.batch_norm")),
        mul_op_(Op::Get("multiply")),
        add_op_(Op::Get("add")) {}

  Expr Rewrite_(const CallNode* n, const Expr& new_n) {
    if (n->op == add_op_ && n->args[0]->IsInstance<CallNode>()) {
      const auto* n1 = n->args[0].as<CallNode>();
      if (n1->op == mul_op_ && n1->args[0]->IsInstance<TupleGetItemNode>()) {
        const auto* n2 = n1->args[0].as<TupleGetItemNode>()->tuple.as<CallNode>();
        if (n2->op == batch_norm_op_) {
          const auto* new_n1 = new_n.as<CallNode>()->args[0].as<CallNode>();
          const auto* new_n2 = new_n1->args[0].as<TupleGetItemNode>()->tuple.as<CallNode>();
          // check whether the constant in mul and add is a 1D tensor
          auto mul_type = n1->args[1]->checked_type().as<TensorTypeNode>();
          size_t mul_ndim = mul_type->shape.size();
          size_t mul_effective_ndim = mul_ndim;
          for (size_t i = 0; i < mul_ndim; i++) {
            if (mul_type->shape[i].as<IntImmNode>()->value == 1) {
              mul_effective_ndim--;
            }
          }
          auto add_type = n->args[1]->checked_type().as<TensorTypeNode>();
          size_t add_ndim = add_type->shape.size();
          size_t add_effective_ndim = add_ndim;
          for (size_t i = 0; i < add_ndim; i++) {
            if (add_type->shape[i].as<IntImmNode>()->value == 1) {
              add_effective_ndim--;
            }
          }
          if (mul_effective_ndim == 1 && add_effective_ndim == 1) {
            return _SimplifyBNMulAdd(new_n2, new_n1->args[1], new_n.as<CallNode>()->args[1]);
          }
        }
      }
    }
    return new_n;
  }

 private:
  // Cache the following ops. They will be used in the passes repeatedly for
  // operator equivalence checking so that the registry lookup overhead can be reduced.
  const Op& batch_norm_op_;
  const Op& mul_op_;
  const Op& add_op_;
};

Expr FoldMulAddToBN(const Expr& e) { return SimplifyBNMulAdd().Mutate(e); }

namespace transform {

Pass FoldMulAddToBN() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldMulAddToBN(f));
      };
  return CreateFunctionPass(pass_func, 0, "FoldMulAddToBN", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldMulAddToBN").set_body_typed(FoldMulAddToBN);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
