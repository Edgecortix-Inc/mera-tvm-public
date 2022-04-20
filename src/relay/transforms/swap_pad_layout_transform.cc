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
 * \file swap_pad_layout_transform.cc
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

class SwapTransformer : public ExprMutator {
 public:
  SwapTransformer() : pad_op_(Op::Get("nn.pad")), layout_transform_op_(Op::Get("layout_transform")) {}

  Expr VisitExpr_(const CallNode* n) {
    if (n->op == layout_transform_op_) {
      auto layout_arg = n->args[0].as<CallNode>();
      if (layout_arg && layout_arg->op == pad_op_) {
        auto pad_arg = layout_arg->args[0];
        auto pad_value = layout_arg->args[1];
        auto new_layout_transform_op =
            Call(layout_transform_op_, {pad_arg}, n->attrs, n->type_args);
        const LayoutTransformAttrs* layout_attrs = n->attrs.as<LayoutTransformAttrs>();
        return TransformPad(layout_arg, layout_attrs->src_layout, layout_attrs->dst_layout,
                            new_layout_transform_op, pad_value, layout_arg->type_args);
      }
    }
    return ExprMutator::VisitExpr_(n);
  }

 private:
  Expr TransformPad(const CallNode* pad, const std::string& src_layout,
                    const std::string& dst_layout, Expr arg, Expr pad_value, const Array<Type>& type_args) {
    const PadAttrs* pad_param = pad->attrs.as<PadAttrs>();
    CHECK(pad_param != nullptr);
    auto nchw_pad_width = pad_param->pad_width;
    auto new_pad_attrs = make_object<PadAttrs>();
    new_pad_attrs->pad_mode = pad_param->pad_mode;
    Array<Array<Integer>> new_pad_width;
    if (src_layout == "NCHW" && dst_layout == "NHWC") {
      new_pad_width.push_back(nchw_pad_width[0]);
      new_pad_width.push_back(nchw_pad_width[2]);
      new_pad_width.push_back(nchw_pad_width[3]);
      new_pad_width.push_back(nchw_pad_width[1]);
    } else if (src_layout == "NHWC" && dst_layout == "NCHW") {
      new_pad_width.push_back(nchw_pad_width[0]);
      new_pad_width.push_back(nchw_pad_width[3]);
      new_pad_width.push_back(nchw_pad_width[1]);
      new_pad_width.push_back(nchw_pad_width[2]);
    } else {
      LOG(FATAL) << "unsupported src and dst layout:" << src_layout << ", " << dst_layout;
    }

    new_pad_attrs->pad_width = new_pad_width;
    return Call(pad_op_, {arg, pad_value}, Attrs(new_pad_attrs), type_args);
  }

  const Op& pad_op_;
  const Op& layout_transform_op_;
};

Expr SwapPadLayoutTransform(const Expr& e) {
  return SwapTransformer().Mutate(e);
}

namespace transform {

Pass SwapPadLayoutTransform() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SwapPadLayoutTransform(f));
      };
  return CreateFunctionPass(pass_func, 0, "SwapPadLayoutTransform", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SwapPadLayoutTransform")
    .set_body_typed(SwapPadLayoutTransform);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
