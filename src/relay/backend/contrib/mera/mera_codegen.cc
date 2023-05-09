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
#include <mera/mdna_compile.h>
#include <mera/mdna_quantize.h>
#include <mera/mdna_ir.h>
#include <mera/mdna_ir_io.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>

#include <algorithm>
#include <string>
#include <vector>

#include "mera_compiler.h"

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"
#include "dlpack/dlpack.h"
#include "tvm/relay/expr.h"
#include "tvm/runtime/data_type.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

std::string GetWeightLayout() {
  auto cfg = transform::PassContext::Current()->GetConfig<MeraCompilerConfig>("relay.ext.mera.options");
  if (!cfg.defined()) {
    cfg = AttrsWithDefaultValues<MeraCompilerConfig>();
  }

  const auto weight_layout = cfg.value()->weight_layout;
  CHECK(weight_layout == "OIHW" || weight_layout == "HWIO")
      << "Weight layout should be either PyTorch or TFLite format";
  return weight_layout;
}

mera::ir::DataType ConvertType(tvm::DataType tvm_dtype) {
  // TODO
  if (tvm_dtype.is_float()) {
    return mera::ir::DataType::Float32;
  } else if (tvm_dtype.is_int() && tvm_dtype.bits() == 32) {
    return mera::ir::DataType::Int32;
  } else if (tvm_dtype.is_int() && tvm_dtype.bits() == 8) {
    return mera::ir::DataType::Int8;
  } else if (tvm_dtype.is_uint() && tvm_dtype.bits() == 8) {
    return mera::ir::DataType::UInt8;
  }

  LOG(FATAL) << "Cannot convert dtype " << tvm_dtype;
  return mera::ir::DataType::Float32;
}

struct Conv2DAttrValues {
  int groups;
  int output_channels;
  mera::ir::Padding pads;
  mera::ir::Strides strides;
  mera::ir::Dilations dilations;
};

Conv2DAttrValues GetConv2DAttrValues(const Conv2DAttrs* conv2d_attr) {
  int groups = conv2d_attr->groups;
  int output_channels = conv2d_attr->channels.as<IntImmNode>()->value;

  int padding_top = conv2d_attr->padding[0].as<IntImmNode>()->value;
  int padding_left = conv2d_attr->padding[1].as<IntImmNode>()->value;
  int padding_bot = conv2d_attr->padding[2].as<IntImmNode>()->value;
  int padding_right = conv2d_attr->padding[3].as<IntImmNode>()->value;

  mera::ir::Padding pads{padding_top, padding_bot, padding_left, padding_right};

  int stride_h = conv2d_attr->strides[0].as<IntImmNode>()->value;
  int stride_w = conv2d_attr->strides[1].as<IntImmNode>()->value;
  mera::ir::Strides strides{stride_h, stride_w};

  int dilation_h = conv2d_attr->dilation[0].as<IntImmNode>()->value;
  int dilation_w = conv2d_attr->dilation[1].as<IntImmNode>()->value;
  mera::ir::Dilations dilations{dilation_h, dilation_w};
  return Conv2DAttrValues{groups, output_channels, pads, strides, dilations};
}

template <typename Attr>
const Attr* GetAttrChecked(const CallNode* call) {
  const auto attr = call->attrs.as<Attr>();
  CHECK(attr);
  return attr;
}

bool IsOp(const CallNode* call, std::string op_name) {
  const auto* op_node = call->op.as<OpNode>();
  CHECK(op_node) << "Expects a single op.";
  Op op = GetRef<Op>(op_node);
  return op == Op::Get(op_name);
}

static std::string mera_arch{};
static std::string mera_ccfg{};

bool IsInputNHWCLayout() {
  auto cfg = transform::PassContext::Current()->GetConfig<MeraCompilerConfig>("relay.ext.mera.options");
  if (!cfg.defined()) {
    cfg = AttrsWithDefaultValues<MeraCompilerConfig>();
  }
  return cfg.value()->input_layout == "NHWC";
}

mera::ir::Shape GetNCHWOutputShape(const mera::ir::Shape& orig_shape, const std::string& input_layout) {
  if (input_layout == "NHWC") {
    CHECK(IsInputNHWCLayout() && orig_shape.rank == 4);
    // Mera IR expects NCHW layout, but this shape is still in NHWC
    // Need to convert output shape from NHWC to NCHW
    auto nchw_shape = {orig_shape.shape[0], orig_shape.shape[3], orig_shape.shape[1],
                       orig_shape.shape[2]};
    return mera::ir::Shape{nchw_shape, orig_shape.rank, orig_shape.size};
  }
  return orig_shape;
}

class Compiler {
 public:
  using Scope = std::map<std::string, mera::ir::Tensor>;

  explicit Compiler(const std::string& id, mera::ir::Module& module)
      : ext_func_id_(id), graph_(module.AddFunction(id)) {}

  void Compile(const Function& func) {
    Scope scope;
    for (auto& param : func->params) {
      auto name(param->name_hint());
      scope[name] = RegisterVar(name, param);
    }
    graph_.AddOutput(Compile(scope, func));
    for (auto& op : graph_.operators) {
      DLOG(INFO) << "MERA_DNA_IR: " << op << std::endl;
    }
  }

 private:
  std::vector<mera::ir::Tensor> Compile(const Scope& current_scope, const RelayExpr& expr) {
    auto it = memo_.find(expr);
    if (it == memo_.end()) {
      auto tensors = CompileExpr(current_scope, expr);
      it = memo_.emplace(expr, tensors).first;
    }
    return it->second;
  }

  void ResolveDwPwConvWeightLayout(const std::string& kernel_id) {
    for (auto& op : graph_.operators) {
      if (op.is<mera::ir::Int8VecConstant>()) {
        auto& c(*op.get<mera::ir::Int8VecConstant>());
        if (c.output.id == kernel_id) {
          std::swap(c.output.shape.shape[0], c.output.shape.shape[1]);
          break;
        }
      }
    }
  }

  std::vector<mera::ir::Tensor> CompileExpr(const Scope& current_scope, const RelayExpr& expr) {
    struct Visitor : public MemoizedExprTranslator<std::vector<mera::ir::Tensor>> {
      Visitor(Compiler& compiler, const Scope& scope, mera::ir::Graph& g)
          : compiler(compiler), current_scope(scope), graph(g) {}

      std::vector<mera::ir::Tensor> VisitExpr_(const FunctionNode* op) {
        for (auto param : op->params) {
          this->VisitExpr(param);
        }
        return VisitExpr(op->body);
      }

      std::vector<mera::ir::Tensor> CompileComposite(const FunctionNode* callee_func,
                                                   const std::string& name,
                                                   const Scope& callee_scope) {
        if (name == "mera.upsampling") {
          const auto* dequantize_call =
              GetRootCall(callee_func->body.as<CallNode>(), 2,
                          std::vector<std::string>{"qnn.dequantize", "image.resize2d", "qnn.quantize"});
          const auto* upsampling_call =
              GetRootCall(callee_func->body.as<CallNode>(), 1, std::vector<std::string>{"image.resize2d", "qnn.quantize"});
          const auto dequantize_args = CompileArgs(dequantize_call, callee_scope);
          const auto& input(dequantize_args[0][0]);
          const auto& input_scale(dequantize_args[1][0]);
          const auto& input_zero_point(dequantize_args[2][0]);
          const auto* upsampling_attr = GetAttrChecked<Resize2DAttrs>(upsampling_call);
          const auto method = upsampling_attr->method;
          std::string node_name;
          if (method == "linear") {
            node_name = "UpsamplingLinear";
          } else if (method == "nearest_neighbor") {
            node_name = "UpsamplingNearest";
          } else {
            CHECK(false) << "Unsupported upsampling method: " << method;
          }
          const auto coord_trans = upsampling_attr->coordinate_transformation_mode;
          const auto shape = GetNCHWOutputShape(GetShape(upsampling_call->checked_type()),
                                                upsampling_attr->layout);
          return {graph.Add<mera::ir::Upsampling>(node_name, input.type, shape, input, input_scale,
                                                input_zero_point, method, coord_trans)};
        } else if (name == "mera.leaky_relu") {
          const auto* dequantize_call =
              GetRootCall(callee_func->body.as<CallNode>(), 2,
                          std::vector<std::string>{"qnn.dequantize", "nn.leaky_relu", "qnn.quantize"});
          const auto* leaky_relu_call =
              GetRootCall(callee_func->body.as<CallNode>(), 1, std::vector<std::string>{"nn.leaky_relu", "qnn.quantize"});
          const auto dequantize_args = CompileArgs(dequantize_call, callee_scope);
          const auto* quantize_call = GetRootCall(callee_func->body.as<CallNode>(), 0, std::vector<std::string>{"qnn.quantize"});
          const auto quantize_args = CompileArgs(quantize_call, callee_scope);
          const auto& input(dequantize_args[0][0]);
          const auto& input_scale(dequantize_args[1][0]);
          const auto& input_zero_point(dequantize_args[2][0]);
          const auto& output_scale(quantize_args[1][0]);
          const auto& output_zero_point(quantize_args[2][0]);
          const auto* leaky_relu_attr = GetAttrChecked<LeakyReluAttrs>(leaky_relu_call);
          return {graph.Add<mera::ir::LeakyReLU>("LeakyReLU", input.type, input.shape, input, input_scale,
              input_zero_point, output_scale, output_zero_point, leaky_relu_attr->alpha)};
        } else if (name == "mera.tflite_silu") {
          const auto* mul_call = GetRootCall(callee_func->body.as<CallNode>(), 0, std::vector<std::string>{"qnn.mul"});
          auto tensors_input = compiler.Compile(callee_scope, mul_call->args[0]);
          auto tensors_lhs_scale = compiler.Compile(callee_scope, mul_call->args[2]);
          auto tensors_lhs_zp = compiler.Compile(callee_scope, mul_call->args[3]);
          auto tensors_rhs_scale = compiler.Compile(callee_scope, mul_call->args[4]);
          auto tensors_rhs_zp = compiler.Compile(callee_scope, mul_call->args[5]);
          auto tensors_out_scale = compiler.Compile(callee_scope, mul_call->args[6]);
          auto tensors_out_zp = compiler.Compile(callee_scope, mul_call->args[7]);
          CHECK_EQ(tensors_input.size(), 1);
          CHECK_EQ(tensors_lhs_scale.size(), 1);
          CHECK_EQ(tensors_lhs_zp.size(), 1);
          CHECK_EQ(tensors_rhs_scale.size(), 1);
          CHECK_EQ(tensors_rhs_zp.size(), 1);
          CHECK_EQ(tensors_out_scale.size(), 1);
          CHECK_EQ(tensors_out_zp.size(), 1);
          const auto& input(tensors_input[0]);
          const auto& input_scale(tensors_lhs_scale[0]);
          const auto& input_zero_point(tensors_lhs_zp[0]);
          const auto& sigmoid_scale(tensors_rhs_scale[0]);
          const auto& sigmoid_zero_point(tensors_rhs_zp[0]);
          const auto& output_scale(tensors_out_scale[0]);
          const auto& output_zero_point(tensors_out_zp[0]);
          return {graph.Add<mera::ir::SiLU>("SiLU", input.type, input.shape, input,
                                          input_scale, input_zero_point,
                                          sigmoid_scale, sigmoid_zero_point,
                                          output_scale, output_zero_point)};
        } else if (name == "mera.tflite_hswish") {
          const auto& call = callee_func->body.as<CallNode>();
          const CallNode* dequantize_call;
          const CallNode* quantize_call;
          const CallNode* req_call = nullptr;
          bool with_req = IsOp(call, "qnn.requantize");
          if (with_req) {
            dequantize_call = GetRootCall(callee_func->body.as<CallNode>(), 4,
              std::vector<std::string>{"qnn.dequantize", "multiply", "divide", "qnn.quantize", "qnn.requantize"});
            quantize_call = GetRootCall(call, 1, std::vector<std::string>{"qnn.quantize", "qnn.requantize"});
            req_call = GetRootCall(call, 0, std::vector<std::string>{"qnn.requantize"});
          } else {
            dequantize_call = GetRootCall(callee_func->body.as<CallNode>(), 3,
              std::vector<std::string>{"qnn.dequantize", "multiply", "divide", "qnn.quantize"});
            quantize_call = GetRootCall(call, 0, std::vector<std::string>{"qnn.quantize"});
          }
          std::vector<mera::ir::Tensor> tensors_out_scale;
          std::vector<mera::ir::Tensor> tensors_out_zp;
          if (with_req) {
            tensors_out_scale = compiler.Compile(callee_scope, req_call->args[3]);
            tensors_out_zp = compiler.Compile(callee_scope, req_call->args[4]);
          } else {
            tensors_out_scale = compiler.Compile(callee_scope, quantize_call->args[1]);
            tensors_out_zp = compiler.Compile(callee_scope, quantize_call->args[2]);
          }
          const auto dequantize_args = CompileArgs(dequantize_call, callee_scope);
          const auto& input(dequantize_args[0][0]);
          const auto& input_scale(dequantize_args[1][0]);
          const auto& input_zero_point(dequantize_args[2][0]);
          const auto& output_scale(tensors_out_scale[0]);
          const auto& output_zero_point(tensors_out_zp[0]);
          const auto ret_type = input.type;
          return {graph.Add<mera::ir::HSwish>("HSwish", ret_type, input.shape, input,
                                            input_scale, input_zero_point,
                                            output_scale, output_zero_point)};
        } else if (name == "mera.qnn_fc") {
          const auto& call = callee_func->body.as<CallNode>();
          const CallNode* req_call;
          const CallNode* bias_call;
          const CallNode* dense_call;
          const CallNode* in_tensor;

          const bool with_cast = IsOp(call, "cast");
          if (!with_cast) {
            req_call = GetRootCall(call, 0, std::vector<std::string>{"qnn.requantize"});
            bias_call = GetRootCall(req_call->args[0].as<CallNode>(), 0, std::vector<std::string>{"nn.bias_add"});
          } else {
            req_call = GetRootCall(call, 2, std::vector<std::string>{"qnn.requantize", "clip", "cast"});
            bias_call = GetRootCall(req_call->args[0].as<CallNode>(), 0, std::vector<std::string>{"nn.bias_add"});
          }
          dense_call = GetRootCall(bias_call->args[0].as<CallNode>(), 0, std::vector<std::string>{"qnn.dense"});
          const bool with_squeeze = IsOp(dense_call->args[0].as<CallNode>(), "squeeze");
          if (with_squeeze) {
            in_tensor = GetRootCall(dense_call->args[0].as<CallNode>(), 1, std::vector<std::string>{"reshape", "squeeze"});
          } else {
            in_tensor = GetRootCall(dense_call->args[0].as<CallNode>(), 2, std::vector<std::string>{"transpose", "reshape", "reshape"});
          }

          auto input_tensor_in = compiler.Compile(callee_scope, in_tensor->args[0]);
          // If weights are supplied unquantized, grab the unquantized values
          const auto* weights_call = dense_call->args[1].as<CallNode>();
          auto weights_tensor_in = (weights_call != nullptr && IsOp(weights_call, "qnn.quantize")) ?
            compiler.Compile(callee_scope, GetRootCall(dense_call->args[1].as<CallNode>(), 0, std::vector<std::string>{"qnn.quantize"})->args[0])
            : compiler.Compile(callee_scope, dense_call->args[1]);
          auto dense_input_zero_point_in = compiler.Compile(callee_scope, dense_call->args[2]);
          auto dense_weight_zero_point_in = compiler.Compile(callee_scope, dense_call->args[3]);
          auto dense_input_scale_in = compiler.Compile(callee_scope, req_call->args[1]);
          auto dense_weight_scale_in = compiler.Compile(callee_scope, dense_call->args[5]);
          auto bias_tensor_in = compiler.Compile(callee_scope, bias_call->args[1]);
          auto output_scale_in = compiler.Compile(callee_scope, req_call->args[3]);
          auto output_zero_point_in = compiler.Compile(callee_scope, req_call->args[4]);

          CHECK_EQ(weights_tensor_in.size(), 1);
          CHECK_EQ(input_tensor_in.size(), 1);
          CHECK_EQ(dense_input_zero_point_in.size(), 1);
          CHECK_EQ(dense_weight_zero_point_in.size(), 1);
          CHECK_EQ(dense_input_scale_in.size(), 1);
          CHECK_EQ(dense_weight_scale_in.size(), 1);
          CHECK_EQ(bias_tensor_in.size(), 1);
          CHECK_EQ(output_scale_in.size(), 1);
          CHECK_EQ(output_zero_point_in.size(), 1);

          const auto& bias(bias_tensor_in[0]);
          const auto& weights(weights_tensor_in[0]);
          const auto& input(input_tensor_in[0]);
          const auto& dense_input_scale(dense_input_scale_in[0]);
          const auto& dense_input_zero_point(dense_input_zero_point_in[0]);
          const auto& dense_weight_scale(dense_weight_scale_in[0]);
          const auto& dense_weight_zero_point(dense_weight_zero_point_in[0]);
          const auto& output_scale(output_scale_in[0]);
          const auto& output_zero_point(output_zero_point_in[0]);

          const auto out_shape1d = GetShape(call->checked_type());
          const mera::ir::Shape out_shape{
            {out_shape1d.shape[0], out_shape1d.shape[1], 1, 1}, 4, out_shape1d.shape[0] * out_shape1d.shape[1]};
          return {graph.Add<mera::ir::Fc>("Fc", input.type, out_shape, input,
                                          weights, dense_input_scale, dense_input_zero_point, dense_weight_scale,
                                          dense_weight_zero_point, bias, output_scale, output_zero_point)};
        } else if (name == "mera.avg_pooling2d") {
          const auto& call = callee_func->body.as<CallNode>();
          const CallNode *in_call;
          const bool with_adaptive = IsOp(GetRootCall(call, 0, std::vector<std::string>{"cast"})->args[0].as<CallNode>(), "nn.adaptive_avg_pool2d");
          if (with_adaptive) {
            in_call = GetRootCall(call, 2, std::vector<std::string>{"cast", "nn.adaptive_avg_pool2d", "cast"});
          } else {
            in_call = GetRootCall(call, 4, std::vector<std::string>{"nn.pad", "nn.pad", "cast", "nn.avg_pool2d", "cast"});
          }

          auto input_tensor_in = compiler.Compile(callee_scope, in_call->args[0]);
          CHECK_EQ(input_tensor_in.size(), 1);
          const auto& input(input_tensor_in[0]);
          const mera::ir::Shape out_shape{
            {input.shape.shape[0], input.shape.shape[1], 1, 1}, 4, input.shape.shape[0] * input.shape.shape[1]
          };
          return {graph.Add<mera::ir::AvgPooling2d>("AvgPooling2d", input.type, out_shape, input)};
        } else if (name == "mera.mean") {
          const auto& call = callee_func->body.as<CallNode>();

          const auto* requantize = GetRootCall(call, 0, std::vector<std::string>{"qnn.requantize"});
          const auto* in_call = GetRootCall(call, 2, std::vector<std::string>{"cast", "mean", "qnn.requantize"});

          auto tensor_in_scale = compiler.Compile(callee_scope, requantize->args[1]);
          auto tensor_in_zp = compiler.Compile(callee_scope, requantize->args[2]);
          auto tensor_out_scale = compiler.Compile(callee_scope, requantize->args[3]);
          auto tensor_out_zp = compiler.Compile(callee_scope, requantize->args[4]);
          auto input_tensor_in = compiler.Compile(callee_scope, in_call->args[0]);

          CHECK_EQ(tensor_in_scale.size(), 1);
          CHECK_EQ(tensor_in_zp.size(), 1);
          CHECK_EQ(tensor_out_scale.size(), 1);
          CHECK_EQ(tensor_out_zp.size(), 1);
          CHECK_EQ(input_tensor_in.size(), 1);

          const auto& tensor_in_scale_v(tensor_in_scale[0]);
          const auto& tensor_in_zp_v(tensor_in_zp[0]);
          const auto& tensor_out_scale_v(tensor_out_scale[0]);
          const auto& tensor_out_zp_v(tensor_out_zp[0]);
          const auto& input(input_tensor_in[0]);

          const mera::ir::Shape out_shape{
            {input.shape.shape[0], input.shape.shape[1], 1, 1}, 4, input.shape.shape[0] * input.shape.shape[1]
          };
          return {graph.Add<mera::ir::Mean>("Mean", input.type,
            out_shape, input, tensor_in_scale_v, tensor_in_zp_v, tensor_out_scale_v, tensor_out_zp_v)};
        }
        return {compiler.Compile(callee_scope, callee_func->body)};
      }

      std::vector<mera::ir::Tensor> VisitExpr_(const CallNode* call) final {
        if (const auto* callee_func = call->op.as<FunctionNode>()) {
          Scope callee_scope;
          CHECK(call->args.size() == callee_func->params.size());
          for (size_t i = 0; i < call->args.size(); ++i) {
            auto tensors = compiler.Compile(current_scope, call->args[i]);
            CHECK_EQ(tensors.size(), 1);
            callee_scope[callee_func->params[i]->name_hint()] = tensors[0];
          }
          auto composite_name = callee_func->GetAttr<runtime::String>(attr::kComposite);
          CHECK(composite_name.defined());
          return CompileComposite(callee_func, composite_name.value(), callee_scope);
        }

        auto flatten_inputs = [](const auto& inputs) {
          std::vector<mera::ir::Tensor> flist;
          for (const auto& input_tuple : inputs) {
            CHECK_EQ(input_tuple.size(), 1);
            flist.push_back(input_tuple[0]);
          }
          return std::move(flist);
        };
        auto input_tuples = CompileArgs(call);
        std::vector<mera::ir::Tensor> flat_inputs;
        if (!IsOp(call, "qnn.concatenate")) {
          flat_inputs = flatten_inputs(input_tuples);
        }
        const auto& inputs(flat_inputs);
        const auto shape = GetShape(call->checked_type());

        if (IsOp(call, "layout_transform")) {
          return {inputs[0]};
        } else if (IsOp(call, "expand_dims")) {
          mera::ir::Tensor out = {inputs[0].type, shape, inputs[0].id};
          return {out};
        } else if (IsOp(call, "nn.relu")) {
          return {graph.Add<mera::ir::ReLU>("ReLU", mera::ir::DataType::Float32, inputs[0].shape,
                                          inputs[0])};
        } else if (IsOp(call, "nn.pad")) {
          const auto* pad_attr = GetAttrChecked<PadAttrs>(call);
          const int padding_top    = pad_attr->pad_width[2][0].as<IntImmNode>()->value;
          const int padding_bottom = pad_attr->pad_width[2][1].as<IntImmNode>()->value;
          const int padding_left   = pad_attr->pad_width[3][0].as<IntImmNode>()->value;
          const int padding_right  = pad_attr->pad_width[3][1].as<IntImmNode>()->value;
          const mera::ir::Padding pad_width{padding_top, padding_bottom, padding_left, padding_right};
          const mera::ir::FloatVecConstant* pad_value_constant = nullptr;
          for (const auto& op : graph.operators) {
            if (op.is<mera::ir::FloatVecConstant>()) {
              auto* c = op.get<mera::ir::FloatVecConstant>();
              if (c->output.id == inputs[1].id) {
                pad_value_constant = c;
              }
            }
          }
          CHECK(pad_value_constant != nullptr);
          const double pad_value = pad_value_constant->values[0];
          return {graph.Add<mera::ir::Pad>("Pad", inputs[0].type, shape, inputs[0], pad_width,
                                         pad_value)};
        } else if (IsOp(call, "clip")) {
          const auto* clip_attr = GetAttrChecked<ClipAttrs>(call);
          float min_value = clip_attr->a_min;
          float max_value = clip_attr->a_max;
          return {graph.Add<mera::ir::Clip>("Clip", inputs[0].type, inputs[0].shape, min_value,
                                          max_value, inputs[0])};
        } else if (IsOp(call, "add")) {
          auto rhs_shape = inputs[1].shape;
          if (rhs_shape.shape.size() == 4 && rhs_shape.size == rhs_shape.shape[1]) {
            // This case happens when the original graph is NHWC
            mera::ir::Shape rhs_new_shape = {{rhs_shape.shape[1]}, 1, rhs_shape.shape[1]};
            mera::ir::Tensor rhs = {inputs[1].type, rhs_new_shape, inputs[1].id};
            return {graph.Add<mera::ir::BiasAdd>("BiasAdd", inputs[0].type, shape, inputs[0], rhs)};
          }
          return {graph.Add<mera::ir::AddOp>("Add", inputs[0].type, shape, inputs[0], inputs[1])};
        } else if (IsOp(call, "nn.conv2d")) {
          const auto* conv2d_attr = GetAttrChecked<Conv2DAttrs>(call);
          const auto attr_values = GetConv2DAttrValues(conv2d_attr);
          const auto& input(inputs[0]);
          const auto& weights(inputs[1]);
          return {graph.Add<mera::ir::Conv2d>("Conv2d", mera::ir::DataType::Float32, shape,
                                            attr_values.dilations, attr_values.pads,
                                            attr_values.strides, attr_values.groups,
                                            attr_values.output_channels, input, weights)};
        } else if (IsOp(call, "qnn.quantize")) {
          const auto& input(inputs[0]);
          const auto output_scale = inputs[1];
          const auto output_zero_point = inputs[2];
          const auto* attr = GetAttrChecked<qnn::QuantizeAttrs>(call);
          const auto dtype = ConvertType(attr->out_dtype);
          return {graph.Add<mera::ir::Quantize>("Quantize", dtype, inputs[0].shape, input,
                                              output_scale, output_zero_point, attr->axis)};
        } else if (IsOp(call, "qnn.dequantize")) {
          const auto& input(inputs[0]);
          const auto& input_scale(inputs[1]);
          const auto& input_zero_point(inputs[2]);
          return {graph.Add<mera::ir::Dequantize>("Dequantize", mera::ir::DataType::Float32,
                                                inputs[0].shape, input, input_scale,
                                                input_zero_point)};
        } else if (IsOp(call, "qnn.conv2d")) {
          const auto* conv2d_attr = GetAttrChecked<Conv2DAttrs>(call);
          const auto attr_values = GetConv2DAttrValues(conv2d_attr);
          auto wt(inputs[1]);
          if (attr_values.groups == 1 && attr_values.output_channels == 1 && wt.shape.shape[0] != attr_values.output_channels) {
            compiler.ResolveDwPwConvWeightLayout(inputs[1].id);
            std::swap(wt.shape.shape[0], wt.shape.shape[1]);
          }
          const auto& input(inputs[0]);
          const auto& weights(wt);
          const auto& input_zero_point(inputs[2]);
          const auto& weight_zero_point(inputs[3]);
          const auto& input_scale(inputs[4]);
          const auto& weight_scale(inputs[5]);
          CHECK_EQ(conv2d_attr->data_layout, "NCHW") << "Inputs expected to be in NCHW layout";
          return {graph.Add<mera::ir::QuantizedConv2d>(
              "QuantizedConv2d", mera::ir::DataType::Int32, shape, attr_values.dilations,
              attr_values.pads, attr_values.strides, attr_values.groups,
              attr_values.output_channels, input, weights, input_scale, input_zero_point,
              weight_scale, weight_zero_point)};
        } else if (IsOp(call, "qnn.requantize")) {
          const auto& data(inputs[0]);
          const auto& input_scale(inputs[1]);
          const auto& input_zero_point(inputs[2]);
          const auto& output_scale(inputs[3]);
          const auto& output_zero_point(inputs[4]);
          const auto* attr = GetAttrChecked<qnn::RequantizeAttrs>(call);
          const auto out_dtype = ConvertType(attr->out_dtype);
          return {graph.Add<mera::ir::Requantize>("Requantize", out_dtype, inputs[0].shape, data,
                                                input_scale, input_zero_point, output_scale,
                                                output_zero_point)};
        } else if (IsOp(call, "qnn.add")) {
          const auto& lhs(inputs[0]);
          const auto& rhs(inputs[1]);
          const auto& lhs_scale(inputs[2]);
          const auto& lhs_zero_point(inputs[3]);
          const auto& rhs_scale(inputs[4]);
          const auto& rhs_zero_point(inputs[5]);
          const auto& output_scale(inputs[6]);
          const auto& output_zero_point(inputs[7]);
          CHECK(lhs.shape.shape == rhs.shape.shape);
          const auto ret_type = lhs.type;
          return {graph.Add<mera::ir::QuantizedAdd>("QuantizedAdd", ret_type, lhs.shape, lhs, rhs,
                                                  lhs_scale, lhs_zero_point, rhs_scale,
                                                  rhs_zero_point, output_scale, output_zero_point)};
        } else if (IsOp(call, "qnn.mul")) {
          const auto& lhs(inputs[0]);
          const auto& rhs(inputs[1]);
          const auto& lhs_scale(inputs[2]);
          const auto& lhs_zero_point(inputs[3]);
          const auto& rhs_scale(inputs[4]);
          const auto& rhs_zero_point(inputs[5]);
          const auto& output_scale(inputs[6]);
          const auto& output_zero_point(inputs[7]);
          CHECK(lhs.shape.shape == rhs.shape.shape);
          const auto ret_type = lhs.type;
          return {graph.Add<mera::ir::QuantizedMul>("QuantizedMul", ret_type, lhs.shape, lhs, rhs,
                                                  lhs_scale, lhs_zero_point, rhs_scale,
                                                  rhs_zero_point, output_scale, output_zero_point)};
        } else if (IsOp(call, "cast")) {
          const auto* attr = GetAttrChecked<CastAttrs>(call);
          const auto dtype = ConvertType(attr->dtype);
          const auto& data(inputs[0]);
          return {graph.Add<mera::ir::Cast>("Cast", dtype, data.shape, data)};
        } else if (IsOp(call, "nn.bias_add")) {
          const auto& data(inputs[0]);
          const auto& bias(inputs[1]);
          return {graph.Add<mera::ir::BiasAdd>("BiasAdd", data.type, inputs[0].shape, data, bias)};
        } else if (IsOp(call, "nn.max_pool2d")) {
          const auto& data(inputs[0]);
          const auto* attr = GetAttrChecked<MaxPool2DAttrs>(call);
          int padding_top = attr->padding[0].as<IntImmNode>()->value;
          int padding_left = attr->padding[1].as<IntImmNode>()->value;
          int padding_bottom = attr->padding[2].as<IntImmNode>()->value;
          int padding_right = attr->padding[3].as<IntImmNode>()->value;
          int strides_h = attr->strides[0].as<IntImmNode>()->value;
          int strides_w = attr->strides[1].as<IntImmNode>()->value;
          int pool_h = attr->pool_size[0].as<IntImmNode>()->value;
          int pool_w = attr->pool_size[1].as<IntImmNode>()->value;
          auto out_shape = GetNCHWOutputShape(shape, attr->layout);
          return {graph.Add<mera::ir::MaxPool2d>("MaxPool2d", data.type, out_shape, data, pool_h,
                                               pool_w, mera::ir::Strides{strides_h, strides_w},
                                               mera::ir::Padding{padding_top, padding_bottom, padding_left, padding_right})};
        } else if (IsOp(call, "nn.leaky_relu")) {
          // this helps mera.leaky_relu to retrieve the output_scale and output_zero_point values.
          return inputs;
        } else if (IsOp(call, "image.resize2d")) {
          const auto* attr = GetAttrChecked<Resize2DAttrs>(call);
          const auto input = inputs[0];
          const auto input_scale = graph.AddFloatVec(std::vector<float>(1, 1.0));
          const auto input_zero_point = graph.AddInt32Vec(std::vector<int32_t>(1, 0));
          const auto method = attr->method;
          std::string node_name;
          if (method == "nearest_neighbor") {
              node_name = "UpsamplingNearest";
          } else {
            CHECK(false) << "Unsupported upsampling method: " << method;
          }
          const auto coord_trans = attr->coordinate_transformation_mode;
          const auto out_shape = GetNCHWOutputShape(shape, attr->layout);
          return {graph.Add<mera::ir::Upsampling>(node_name, input.type, out_shape, input,
            input_scale, input_zero_point, method, coord_trans)};
        } else if (IsOp(call, "qnn.concatenate")) {
          const auto& data(input_tuples[0]);
          const auto& input_scales(input_tuples[1]);
          const auto& input_zps(input_tuples[2]);
          const auto& output_scale(input_tuples[3][0]);
          const auto& output_zero_point(input_tuples[4][0]);
          const auto* attr = GetAttrChecked<ConcatenateAttrs>(call);
          int axis = attr->axis;
          const auto& dtype = data[0].type;
          int nchw_depth_sum = 0;
          CHECK(data.size() >= 1);
          for (const auto& it : data) {
            nchw_depth_sum += it.shape.shape.at(1);
          }

          // Apply requantizations of the input tensors to output domain
          CHECK_EQ(data.size(), input_scales.size());
          CHECK_EQ(data.size(), input_zps.size());
          std::vector<mera::ir::Tensor> in_tensors;
          for (size_t i = 0; i < data.size(); ++i) {
            const auto req = graph.Add<mera::ir::Requantize>("Requantize", data[i].type, data[i].shape,
              data[i], input_scales[i], input_zps[i], output_scale, output_zero_point);
            in_tensors.emplace_back(req);
          }

          // Inherit the shape from the input with the channels of the output
          const mera::ir::Shape out_shape{
            {data[0].shape.shape[0], nchw_depth_sum, data[0].shape.shape[2], data[0].shape.shape[3]}, 4,
            data[0].shape.shape[0] * nchw_depth_sum * data[0].shape.shape[2] * data[0].shape.shape[3]};
          return {graph.Add<mera::ir::Concatenate>("Concatenate", dtype, out_shape, in_tensors, axis)};
        } else {
          LOG(FATAL) << "Compiler Unsupported operator: " << AsText(call->op, false);
        }
        return {};
      }

      std::vector<mera::ir::Tensor> VisitExpr_(const VarNode* var) final {
        return {current_scope.at(var->name_hint())};
      }

      std::vector<mera::ir::Tensor> VisitExpr_(const ConstantNode* constant) final {
        const auto shape = constant->data.Shape();
        std::vector<int> int_shape(shape.size());
        const auto weight_layout = GetWeightLayout();

        auto convert_layout = [&](const auto& values) {
          if (int_shape.size() == 4 && weight_layout == "HWIO") {
            const auto H = int_shape[0];
            const auto W = int_shape[1];
            const auto I = int_shape[2];
            const auto O = int_shape[3];
            auto ret(values);
            for (int o = 0; o < O; ++o) {
              for (int i = 0; i < I; ++i) {
                for (int h = 0; h < H; ++h) {
                  for (int w = 0; w < W; ++w) {
                    ret[w + h * W + i * H * W + o * I * H * W] =
                        values[o + i * O + w * I * O + h * I * O * W];
                  }
                }
              }
            }
            return ret;
          }
          return values;
        };

        auto convert_shape = [&](const std::vector<int>& shape) {
          if (int_shape.size() == 4 && weight_layout == "HWIO" && shape[3] == 1 && shape[2] > 1) {
            // TFLite depth wise conv weight has shape like (3, 3, 96, 1)
            // Convert it to (96, 1, 3, 3) to be consistent with PyTorch layout
            std::vector<int> new_shape{shape[2], shape[3], shape[0], shape[1]};
            return new_shape;
          } else if (int_shape.size() == 4 && weight_layout == "HWIO") {
            std::vector<int> new_shape{shape[3], shape[2], shape[0], shape[1]};
            return new_shape;
          }
          return shape;
        };

        for (size_t i = 0; i < int_shape.size(); ++i) int_shape[i] = static_cast<int>(shape[i]);
        const int elements = std::accumulate(int_shape.begin(), int_shape.end(), 1,
                                             [](auto acc, auto val) { return acc * val; });

        const auto mera_shape = mera::ir::Shape{convert_shape(int_shape), int(shape.size()), elements};

        if (constant->data->dtype.code == kDLFloat && constant->data->dtype.bits == 32) {
          const auto* ptr = static_cast<float*>(constant->data->data);
          const std::vector<float> values(ptr, ptr + elements);
          return {graph.Add<mera::ir::FloatVecConstant>("FloatConstant", mera::ir::DataType::Float32,
                                                      mera_shape, convert_layout(values))};
        } else if (constant->data->dtype.code == kDLInt && constant->data->dtype.bits == 32) {
          const auto* ptr = static_cast<int*>(constant->data->data);
          const std::vector<int> values(ptr, ptr + elements);
          return {graph.Add<mera::ir::Int32VecConstant>("IntConstant", mera::ir::DataType::Int32,
                                                      mera_shape, convert_layout(values))};
        } else if (constant->data->dtype.code == kDLInt && constant->data->dtype.bits == 8) {
          const auto* ptr = static_cast<int8_t*>(constant->data->data);
          const std::vector<int8_t> values(ptr, ptr + elements);
          return {graph.Add<mera::ir::Int8VecConstant>("IntConstant", mera::ir::DataType::Int8,
                                                     mera_shape, convert_layout(values))};
        } else {
          LOG(FATAL) << "Unsupported constant type";
        }
        return {};
      }

      std::vector<mera::ir::Tensor> VisitExpr_(const TupleNode* tup) final {
        std::vector<mera::ir::Tensor> results;
        for (auto item : tup->fields) {
          auto tensors = compiler.Compile(current_scope, item);
          CHECK_EQ(tensors.size(), 1);
          results.push_back(tensors[0]);
        }
        return results;
      }

      std::vector<std::vector<mera::ir::Tensor>> CompileArgs(const CallNode* call, const Scope& scope) {
        std::vector<std::vector<mera::ir::Tensor>> arg_tensors;
        for (auto& arg : call->args) {
          auto tensors = compiler.Compile(scope, arg);
          arg_tensors.emplace_back(tensors);
        }
        return arg_tensors;
      }

      std::vector<std::vector<mera::ir::Tensor>> CompileArgs(const CallNode* call) {
        return CompileArgs(call, current_scope);
      }

      Compiler& compiler;
      const Scope& current_scope;
      mera::ir::Graph& graph;
    };
    Visitor visitor(*this, current_scope, graph_);
    return visitor.VisitExpr(expr);
  }

  mera::ir::Tensor RegisterVar(const std::string& name, const relay::Var& var) {
    const auto type = var->checked_type();
    auto shape = GetShape(type);
    if (IsInputNHWCLayout()) {
      auto nchw_shape = {shape.shape[0], shape.shape[3], shape.shape[1], shape.shape[2]};
      shape.shape = nchw_shape;
    }
    return graph_.Add<mera::ir::Var>(name, GetType(type), shape);
  }

 private:
  std::string ext_func_id_;
  std::unordered_map<Expr, std::vector<mera::ir::Tensor>, ObjectHash, ObjectEqual> memo_;
  mera::ir::Graph& graph_;
};

IRModule TransformLayoutToNCHW(const IRModule& mod) {
  Map<String, Array<String>> desired_layouts;
  Array<String> layouts = {"NCHW", "default"};
  desired_layouts.Set("qnn.conv2d", layouts);
  auto layout_transform_pass = transform::ConvertLayout(desired_layouts);
  auto swap_pad_layout_transform_pass = transform::SwapPadLayoutTransform();
  return swap_pad_layout_transform_pass(layout_transform_pass(mod));
}

class MeraModuleCodeGen {
 public:
  void GenECFunc(const Function& func) {
    CHECK(func.defined()) << "Error: expected a Relay function";
    auto sid = GetExtSymbol(func);
    Compiler c(sid, module_);
    c.Compile(func);
  }

  runtime::Module CompileModule(const ObjectRef& ref) {
    CHECK(ref->IsInstance<FunctionNode>());
    auto func = Downcast<Function>(ref);
    auto func_name = GetExtSymbol(func);

    auto mod = IRModule::FromExpr(Downcast<Function>(ref));

    // NHWC -> NCHW conversion
    // This is required because Mera codegen demands a NCHW graph
    if (IsInputNHWCLayout()) {
      mod = TransformLayoutToNCHW(mod);
    }

    for (const auto& it : mod->functions) {
      GenECFunc(Downcast<Function>(it.second));
    }

    // call the Mera Compiler
    std::vector<uint8_t> code = mera::compile::Compile(module_, mera_arch, mera_ccfg);

    DLTensor input;
    input.data = code.data();
    input.device = DLDevice{kDLCPU, 0};
    input.ndim = 1;
    input.dtype = DLDataType{kDLUInt, 8, 1};
    int64_t shape[] = {int64_t(code.size())};
    input.shape = shape;
    input.strides = nullptr;
    input.byte_offset = 0;
    bool interpreter = mera_arch.empty() ? true : false;
    const auto* pf = runtime::Registry::Get("runtime.module.mera_module_create_empty");
    return (*pf)(&input, interpreter, func_name);
  }

  std::string GetExtSymbol(const Function& func) const {
    const auto name_node = func->GetAttr<tvm::runtime::String>(tvm::attr::kGlobalSymbol);
    CHECK(name_node.defined()) << "Fail to retrieve external symbol.";
    return name_node.value();
  }

 private:
  mera::ir::Module module_;
};

runtime::Module CompileModuleFp32(const ObjectRef &ref) {
  CHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  auto mod = IRModule::FromExpr(func);
  auto cdgen_cfg = GetMeraCompilerConfig();

  CHECK_EQ(mod->functions.size(), 1);
  const auto f_compile = Downcast<Function>((*mod->functions.begin()).second);
  CHECK(func.defined()) << "Expected a Relay function";

  mera::ir::Module mera_mod;
  MeraFp32Compiler codegen(GetExtSymbol(f_compile), mera_mod);
  codegen.Compile(f_compile);

  // Call MERA compiler
  std::vector<uint8_t> code = mera::compile::Compile(mera_mod, cdgen_cfg->mera_arch, cdgen_cfg->mera_ccfg);

  DLTensor input;
  input.data = code.data();
  input.device = DLDevice{kDLCPU, 0};
  input.ndim = 1;
  input.dtype = DLDataType{kDLUInt, 8, 1};
  int64_t shape[] = {int64_t(code.size())};
  input.shape = shape;
  input.strides = nullptr;
  input.byte_offset = 0;

  if (cdgen_cfg->mera_target == "Quantizer") {
    const auto* pf = runtime::Registry::Get("runtime.module.mera_quantizer_create");
    CHECK_NOTNULL(pf);
    return (*pf)(&input, func_name);
  } else if (cdgen_cfg->mera_target == "Interpreter" || cdgen_cfg->mera_target == "InterpreterHwBf16") {
    const auto* pf = runtime::Registry::Get("runtime.module.mera_module_create_empty");
    CHECK_NOTNULL(pf);
    return (*pf)(&input, true, func_name);
  } else {
    CHECK(false) << "Unknown Interpreter target for fp32 MERA: '" << cdgen_cfg->mera_target << "'";
    return runtime::Module{};
  }
}

runtime::Module CompileModuleQuantized(const ObjectRef &ref) {
  CHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  CHECK(func.defined()) << "Expected a Relay function";
  auto func_name = GetExtSymbol(func);
  auto mod = IRModule::FromExpr(func);
  auto cdgen_cfg = GetMeraCompilerConfig();

  CHECK_EQ(mod->functions.size(), 1);
  const auto f_compile = Downcast<Function>((*mod->functions.begin()).second);

  const auto *res_root = f_compile.as<FunctionNode>();
  CHECK_NOTNULL(res_root);
  const auto *res_call = res_root->body.as<CallNode>();
  CHECK_NOTNULL(res_call);
  CHECK(res_call->op.as<OpNode>() && res_call->op.as<OpNode>()->name == "mera_quantized_result")
    << "Relay is not a MERA quantized result";
  const auto *mera_data = res_call->args[1].as<ConstantNode>();
  CHECK_NOTNULL(mera_data);
  CHECK_EQ(mera_data->data.Shape().size(), 1);
  std::vector<uint8_t> data_raw(mera_data->data.Shape().at(0));
  mera_data->data.CopyToBytes(data_raw.data(), data_raw.size());

  const auto mera_mod = mera::quantizer::LoadMeraQuantizedModule(data_raw, func_name);
  // Call MERA compiler
  std::vector<uint8_t> code = mera::compile::Compile(mera_mod, cdgen_cfg->mera_arch, cdgen_cfg->mera_ccfg);

  DLTensor input;
  input.data = code.data();
  input.device = DLDevice{kDLCPU, 0};
  input.ndim = 1;
  input.dtype = DLDataType{kDLUInt, 8, 1};
  int64_t shape[] = {int64_t(code.size())};
  input.shape = shape;
  input.strides = nullptr;
  input.byte_offset = 0;
  bool interpreter = mera_arch.empty();
  const auto* pf = runtime::Registry::Get("runtime.module.mera_module_create_empty");
  return (*pf)(&input, interpreter, func_name);
}

runtime::Module MeraCompiler(const ObjectRef& ref) {
  MeraModuleCodeGen c;
  return c.CompileModule(ref);
}
TVM_REGISTER_GLOBAL("relay.ext.mera").set_body_typed(MeraCompiler);

TVM_REGISTER_GLOBAL("relay.ext.mera_fp32").set_body_typed(CompileModuleFp32);

TVM_REGISTER_GLOBAL("relay.ext.mera_qtz").set_body_typed(CompileModuleQuantized);

void SetMeraArch(const std::string& arch) { mera_arch = arch; }
TVM_REGISTER_GLOBAL("relay.ext.mera.set_arch").set_body_typed(SetMeraArch);

void SetMeraCConfig(const std::string& ccfg) { mera_ccfg = ccfg; }
TVM_REGISTER_GLOBAL("relay.ext.mera.set_ccfg").set_body_typed(SetMeraCConfig);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
