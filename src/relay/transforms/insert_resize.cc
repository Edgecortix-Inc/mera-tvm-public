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
 * \file insert_resize.cc
 */
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

class InsertResizeTransformer : public ExprMutator {
 public:
  InsertResizeTransformer(int new_in_height, int new_in_width)
      : qnn_quantize_op_(Op::Get("qnn.quantize")),
        new_in_height_(new_in_height),
        new_in_width_(new_in_width) {}

  Expr VisitExpr_(const FunctionNode* n) {
    CHECK(n->params.size() == 1) << "Expects a main function with a single input";
    new_input_var_ = MakeNewInputVar(n->params[0]);
    auto new_body = ExprMutator::VisitExpr(n->body);
    return Function({new_input_var_}, new_body, n->ret_type, {});
  }

  Expr VisitExpr_(const CallNode* n) {
    if (n->op == qnn_quantize_op_) {
      auto quantized_input = n->args[0];
      if (quantized_input->IsInstance<VarNode>()) {
        auto var = Downcast<Var>(quantized_input);
        auto tensor_type = Downcast<TensorType>(var->type_annotation);
        Array<PrimExpr> out_size;
        // Assume NHWC layout!!
        out_size.push_back(tensor_type->shape[1]);
        out_size.push_back(tensor_type->shape[2]);
        const auto qattrs = n->attrs.as<qnn::QuantizeAttrs>();
        const auto axis = qattrs->axis;
        const auto input_scale = n->args[1];
        const auto input_zero_point = n->args[2];
        auto resize_input =
            MakeNewInputQuantize(axis, input_scale, input_zero_point, qattrs->out_dtype);
        return MakeQuantizedResize(resize_input, out_size, axis, input_scale, input_zero_point,
                                   qattrs->out_dtype);
      }
    }
    return ExprMutator::VisitExpr_(n);
  }

 private:
  Var MakeNewInputVar(Var orig_var) {
    auto tensor_type = Downcast<TensorType>(orig_var->type_annotation);
    Array<PrimExpr> new_shape;
    new_shape.push_back(tensor_type->shape[0]);
    new_shape.push_back(new_in_height_);
    new_shape.push_back(new_in_width_);
    new_shape.push_back(tensor_type->shape[3]);
    auto new_tensor_type = TensorType(new_shape, tensor_type->dtype);
    return Var(orig_var->name_hint(), new_tensor_type);
  }

  Expr MakeNewInputQuantize(int axis, Expr input_scale, Expr input_zero_point, DataType out_dtype) {
    auto quant_attrs = make_object<qnn::QuantizeAttrs>();
    quant_attrs->out_dtype = out_dtype;
    quant_attrs->axis = axis;
    CHECK(new_input_var_.defined());
    return Call(Op::Get("qnn.quantize"), {new_input_var_, input_scale, input_zero_point},
                Attrs(quant_attrs), {});
  }

  Expr MakeQuantizedResize(Expr input, const Array<PrimExpr>& out_size, int axis, Expr input_scale,
                           Expr input_zero_point, DataType out_dtype) {
    auto dequant_attrs = make_object<qnn::DequantizeAttrs>();
    auto resize_attrs = MakeResizeAttrs(out_size);
    auto quant_attrs = make_object<qnn::QuantizeAttrs>();
    dequant_attrs->axis = axis;
    quant_attrs->out_dtype = out_dtype;
    quant_attrs->axis = axis;
    auto dequant = Call(Op::Get("qnn.dequantize"), {input, input_scale, input_zero_point},
                        Attrs(dequant_attrs), {});
    auto resize = Call(Op::Get("image.resize2d"), {dequant}, Attrs(resize_attrs), {});
    return Call(Op::Get("qnn.quantize"), {resize, input_scale, input_zero_point},
                Attrs(quant_attrs), {});
  }

  ObjectPtr<Resize2DAttrs> MakeResizeAttrs(const Array<PrimExpr>& out_size) {
    auto resize_attrs = make_object<Resize2DAttrs>();
    resize_attrs->layout = "NHWC";
    resize_attrs->method = "bilinear";
    resize_attrs->coordinate_transformation_mode = "align_corners";
    resize_attrs->size = out_size;
    return resize_attrs;
  }

  const Op& qnn_quantize_op_;
  int new_in_height_;
  int new_in_width_;
  Var new_input_var_;
};

Expr InsertResize(const Expr& e, int new_in_height, int new_in_width) {
  return InsertResizeTransformer(new_in_height, new_in_width).Mutate(e);
}

namespace transform {

Pass InsertResize(int new_in_height, int new_in_width) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto main_func = m->Lookup("main");
        if (main_func == f) {
          return Downcast<Function>(InsertResize(f, new_in_height, new_in_width));
        }
        return f;
      };
  return CreateFunctionPass(pass_func, 0, "InsertResize", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.InsertResize").set_body_typed(InsertResize);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
