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
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/image.h>
#include "mera_compiler.h"

namespace tvm::relay::contrib {

TVM_REGISTER_NODE_TYPE(MeraCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.mera.options", MeraCompilerConfig);
TVM_REGISTER_NODE_TYPE(MeraQtzCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.mera_qtz.options", MeraQtzCompilerConfig);

bool IRContext::IRTraverse::IsOp(const std::string &op_name) const {
  const auto *op_node = curr_ir_pos->op.as<OpNode>();
  CHECK_NOTNULL(op_node);
  return op_node->name == op_name; 
}

IRContext::IRTraverse IRContext::IRTraverse::Get(unsigned index) {
  CHECK(curr_ir_pos->args.size() > index);
  const CallNode *new_pos = curr_ir_pos->args[index].as<CallNode>();
  CHECK_NOTNULL(new_pos);
  return IRTraverse(new_pos, owner);
}

bool IRContext::IRTraverse::HasCall(unsigned index) {
  CHECK(curr_ir_pos->args.size() > index);
  return curr_ir_pos->args[index].as<CallNode>() != nullptr;
}

IRContext::IRTraverse IRContext::IRTraverse::MoveIf(const std::string &opt_op_name, unsigned index) {
  return IsOp(opt_op_name) ? Get(index) : *this;
}

static inline mera::ir::Shape GetShapeNchw(const IRContext &ir) {
  return GetShapeNchw(ir.GetRootCall());
}

static inline mera::ir::Shape GetShape(const IRContext &ir) {
  return GetShape(ir.GetRootCall());
}

/**
 * @brief Parses the output shape and, if the tensor is 4D, converts it to NCHW
 */
static inline mera::ir::Shape GetShapeNchwMultiDim(const IRContext &ir) {
  auto shape = GetShape(ir.GetRootCall());
  if (shape.rank == 4) {
    return GetShapeNchw(ir.GetRootCall());
  }
  return shape;
}

MeraCompilerConfig GetMeraCompilerConfig() {
  auto cfg = transform::PassContext::Current()->GetConfig<MeraCompilerConfig>("relay.ext.mera.options");
  return cfg.value_or(AttrsWithDefaultValues<MeraCompilerConfig>());
}

MeraQtzCompilerConfig GetMeraQtzCompilerConfig() {
  auto cfg = transform::PassContext::Current()->GetConfig<MeraQtzCompilerConfig>("relay.ext.mera_qtz.options");
  return cfg.value_or(AttrsWithDefaultValues<MeraQtzCompilerConfig>());
}

struct Conv2DAttrValues {
  int groups;
  int output_channels;
  mera::ir::Padding pads;
  mera::ir::Strides strides;
  mera::ir::Dilations dilations;
};

template<typename ConvAttrT = Conv2DAttrs>
Conv2DAttrValues GetConv2DAttrValues(const CallNode *call) {
  const auto attr = AsChecked<ConvAttrT>(call->attrs);
  Conv2DAttrValues res;
  res.groups = attr->groups;
  res.output_channels = GetInt(attr->channels);
  res.pads = AttrToMeraPadding(attr->padding);
  res.strides = AttrToMeraStrides(attr->strides);
  res.dilations = AttrToMeraDilations(attr->dilation);
  return res;
}

struct MeraCompilerVisitor : public backend::MemoizedExprTranslator<TensorVec_t> {
  TensorVec_t VisitExpr_(const FunctionNode *op) final {
    // A function node can only be called from the top level as composites are functions flattened on the MERA IR
    // Thus this call should not happen from within the visitor for generating IR
    LOG(FATAL) << "Unexpected call to visitor with FunctionNode";
    return {};
  }

  TensorVec_t VisitExpr_(const CallNode* call) final {
    // We process calls as top level composite functions only
    const auto *func_call = call->op.as<FunctionNode>();
    CHECK_NOTNULL(func_call);
    TensorVec_t input_tensors;
    CHECK(call->args.size() == func_call->params.size());
    for (unsigned i = 0; i < call->args.size(); ++i) {
      auto compile_tensors = this->VisitExpr(call->args[i]);
      CHECK_EQ(compile_tensors.size(), 1);
      input_tensors.emplace_back(compile_tensors[0]);
    }
    auto composite_name = func_call->GetAttr<runtime::String>(attr::kComposite);
    CHECK(composite_name.defined());
    auto it_func = compiler.codegen_funcs_.find(composite_name.value());
    CHECK(it_func != compiler.codegen_funcs_.end()) << "Could not find a valid implementation for Composite '"
      << composite_name.value() << "' in the compiler.";
    const auto *call_body = func_call->body.as<CallNode>();
    CHECK_NOTNULL(call_body);
    IRContext ir{this, call_body};
    return it_func->second(input_tensors, ir);
  }

  TensorVec_t VisitExpr_(const VarNode* var) final {
    return {current_scope.at(var->name_hint())};
  }

  template<typename T>
  std::vector<T> ProcessConstant(void *data, mera::ir::Shape &shape) {
    T *ptr = static_cast<T*>(data);
    if (parse_mode == COPY || parse_mode == FLATTEN_W) {
      if (parse_mode == FLATTEN_W) {
        shape = mera::ir::Shape{{shape.size}, mera::ir::layout::W};
      }
      return std::vector<T>(ptr, ptr + shape.size);
    } else if (parse_mode == WEIGHT_SWAP_LAYOUT) {
      // Convert from HWIO to OIHW
      CHECK_EQ(shape.rank, 4);
      const auto H = shape.shape[0];
      const auto W = shape.shape[1];
      const auto I = shape.shape[2];
      const auto O = shape.shape[3];
      std::vector<T> ret(shape.size);
      for (int o = 0; o < O; ++o) {
        for (int i = 0; i < I; ++i) {
          for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
              ret[w + h * W + i * H * W + o * I * H * W] =
                ptr[o + i * O + w * I * O + h * I * O * W];
            }
          }
        }
      }
      shape.shape = {O, I, H, W};
      shape.layout = mera::ir::layout::OIHW;
      return ret;
    } else {
      LOG(FATAL) << "Unsupported constant parse mode " << parse_mode;
    }
    return {};
  }

  TensorVec_t VisitExpr_(const ConstantNode* constant) final {
    auto shape = GetShape(constant->checked_type());
    const auto mera_type = GetType(constant->checked_type());
    void *constant_data = constant->data->data;
    if (mera_type == mera::ir::DataType::Float32) {
      const std::vector<float> values = ProcessConstant<float>(constant_data, shape);
      return TensorVec_t{graph.Add<mera::ir::FloatVecConstant>("FloatConstant", mera_type, shape, values)};
    } else if (mera_type == mera::ir::DataType::Int32) {
      const std::vector<int32_t> values = ProcessConstant<int32_t>(constant_data, shape);
      return TensorVec_t{graph.Add<mera::ir::Int32VecConstant>("Int32Constant", mera_type, shape, values)};
    } else if (mera_type == mera::ir::DataType::Int8 || mera_type == mera::ir::DataType::UInt8) {
      const std::vector<int8_t> values = ProcessConstant<int8_t>(constant_data, shape);
      return TensorVec_t{graph.Add<mera::ir::Int8VecConstant>("Int8Constant", mera_type, shape, values)};
    } else {
      LOG(FATAL) << "Unsupported constant type";
      return {};
    }
  }

  TensorVec_t VisitExpr_(const TupleNode* tup) final {
    TensorVec_t res;
    for (const auto &t_field : tup->fields) {
      auto t = VisitExpr(t_field);
      CHECK_EQ(t.size(), 1);
      res.push_back(t[0]);
    }
    return res;
  }

  TensorVec_t VisitExpr_(const TupleGetItemNode* tup) final {
    const auto* tuple = tup->tuple.as<TupleNode>();
    return VisitExpr(tuple->fields[tup->index]);
  }

  MeraCompilerBase &compiler;
  const Scope_t& current_scope;
  mera::ir::Graph& graph;

  constant_parse_mode_t parse_mode = COPY;

  MeraCompilerVisitor(MeraCompilerBase &compiler, const Scope_t &scope, mera::ir::Graph &graph):
    compiler(compiler), current_scope(scope), graph(graph) {}
};

void MeraCompilerBase::Compile(const tvm::relay::Function &func) {
  Scope_t scope;
  // Add all Vars
  for (auto &param : func->params) {
    const auto name(param->name_hint());
    const auto type(param->checked_type());
    auto shape = GetShape(type);
    if (shape.rank == 4) {
      // Auto-convert 4D tensors to NCHW
      shape = GetShapeNchw(type);
    }
    scope[name] = graph_.Add<mera::ir::Var>(name, GetType(type), shape);
  }

  const auto *mod_main = func.as<FunctionNode>();
  CHECK_NOTNULL(mod_main);
  MeraCompilerVisitor visitor(*this, scope, graph_);
  graph_.AddOutput(visitor.VisitExpr(mod_main->body));
}

mera::ir::Tensor IRContext::IRTraverse::CompileConstant(unsigned arg_idx, constant_parse_mode_t parse_mode) const {
  CHECK(curr_ir_pos->args.size() > arg_idx);
  CHECK_NOTNULL(curr_ir_pos->args[arg_idx].as<ConstantNode>());

  auto *visitor = dynamic_cast<MeraCompilerVisitor*>(owner.visitor);
  CHECK_NOTNULL(visitor);
  visitor->parse_mode = parse_mode;
  auto tensors = visitor->VisitExpr(curr_ir_pos->args[arg_idx]);
  visitor->parse_mode = COPY;
  CHECK_EQ(tensors.size(), 1);
  return tensors[0];
}

static TensorVec_t EmitAttentionNode(const std::vector<mera::ir::Tensor> &inputs, IRContext &ir, mera::ir::Graph &graph,
    bool const_query = false, bool has_mask = false) {
  auto out_shape = GetShape(ir);
  const int embed_dim = out_shape.shape[out_shape.rank - 1];
  const int batch = out_shape.rank == 3 ? out_shape.shape[0] : 1;
  const int query_length = out_shape.rank == 3 ? out_shape.shape[1] : out_shape.shape[0];

  auto itt = ir.Traverse()[0][0][0];
  CHECK(itt.IsOp("nn.batch_matmul"));
  const auto val_shape = GetShape(itt.GetCall()->args[1].as<CallNode>()->checked_type());
  CHECK_EQ(val_shape.rank, 3);
  // val_shape will have the form of [B * num_heads, dim, L]
  CHECK(val_shape.shape[0] % batch == 0);
  const int num_heads = val_shape.shape[0] / batch;
  const int dim = val_shape.shape[1];
  const int seq_length = val_shape.shape[2];
  CHECK_EQ(num_heads * dim, embed_dim) << "embed_dim(" << embed_dim << ") must be a multiple of dim_x(" << dim
    << ") times num_heads(" << num_heads << ")";

  int slice_value, slice_query, slice_key;
  mera::ir::Tensor query_tensor, key_tensor, value_tensor;
  if (const_query) {
    CHECK_EQ(inputs.size(), 2); // Query is a constant
    // In this case value and key should come from the same tensor
    CHECK_EQ(inputs[0].id, inputs[1].id);
    std::tie(slice_query, slice_key, slice_value) = std::make_tuple(0, 0, 1);
    // Search for first matmul of constant query with key
    auto itt = ir.Traverse()[0][0][0][0][0][0];
    CHECK(itt.IsOp("nn.batch_matmul"));
    query_tensor = itt.CompileConstant(0);
    key_tensor = inputs[0];
    value_tensor = inputs[1];
  } else if (inputs.size() == 3) {
    // Source are 3 distinct var connections (even if they come from the same one)
    typedef enum { QUERY = 0, KEY = 1, VALUE = 2 } attn_idx_t;
    const std::set<std::string> source_inputs{inputs[KEY].id, inputs[QUERY].id, inputs[VALUE].id};
    if (source_inputs.size() == 1) {
      // Case when all three inputs come from same tensor: assign slices in Q,K,V order
      std::tie(slice_query, slice_key, slice_value) = std::make_tuple(0, 1, 2);
    } else if (source_inputs.size() == 2) {
      // We support the case of V and K from same source as part of decoder network
      CHECK_EQ(inputs[VALUE].id, inputs[KEY].id);
      std::tie(slice_query, slice_key, slice_value) = std::make_tuple(0, 0, 1);
    } else {
      // Inputs are all different tensors, don't slice the inputs
      std::tie(slice_query, slice_key, slice_value) = std::make_tuple(0, 0, 0);
    }
    query_tensor = inputs[QUERY];
    key_tensor = inputs[KEY];
    value_tensor = inputs[VALUE];
  } else if (inputs.size() == 1) {
    // Source is a single input: assign slices in Q,K,V order
    std::tie(slice_query, slice_key, slice_value) = std::make_tuple(0, 1, 2);
    query_tensor = inputs[0];
    key_tensor = inputs[0];
    value_tensor = inputs[0];
  } else {
    LOG(FATAL) << "Unsupported attention source case.";
  }

  // Reorder input tensors Q,K,V to MERA IR V,Q,K
  return TensorVec_t{graph.Add<mera::ir::Attention>("Attention", mera::ir::DataType::Float32, out_shape,
    value_tensor, query_tensor, key_tensor, dim, num_heads, seq_length, query_length,
    slice_value, slice_query, slice_key, has_mask, const_query)};
}

/*
 * FP32 COMPILER
 */

MeraFp32Compiler::MeraFp32Compiler(const std::string &id, mera::ir::Module &module):
  MeraCompilerBase(id, module, {
    {"mera_fp32.conv2d", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto attr = GetConv2DAttrValues(ir.GetRootCall());
      const auto weight = ir.Traverse().CompileConstant(1, WEIGHT_SWAP_LAYOUT);
      return TensorVec_t{graph_.Add<mera::ir::Conv2d>("Conv2d", kType, GetShapeNchw(ir),
        attr.dilations, attr.pads, attr.strides, attr.groups,
        attr.output_channels, inputs[0], weight)};
    }},
    {"mera_fp32.tconv2d", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto attr = GetConv2DAttrValues<Conv2DTransposeAttrs>(ir.GetRootCall());
      const auto weight = ir.Traverse().CompileConstant(1, WEIGHT_SWAP_LAYOUT);
      return TensorVec_t{graph_.Add<mera::ir::TransConv2d>("TransConv2d", kType, GetShapeNchw(ir),
        attr.dilations, attr.pads, attr.strides, attr.groups,
        attr.output_channels, inputs[0], weight)};
    }},
    {"mera_fp32.bias_add", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto bias = ir.Traverse().CompileConstant(1);
      return TensorVec_t{graph_.Add<mera::ir::BiasAdd>("BiasAdd", kType, GetShapeNchw(ir),
        inputs[0], bias)};
    }},
    {"mera_fp32.relu", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::ReLU>("ReLU", kType, GetShapeNchwMultiDim(ir), inputs[0])};
    }},
    {"mera_fp32.leaky_relu", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto *attr = AsChecked<LeakyReluAttrs>(ir.GetRootCall()->attrs);
      return TensorVec_t{graph_.Add<mera::ir::LeakyReLUFp>("LeakyReLU", kType, GetShapeNchw(ir), inputs[0], attr->alpha)};
    }},
    {"mera_fp32.silu", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::SiLUFp>("SiLU", kType, GetShapeNchwMultiDim(ir), inputs[0])};
    }},
    {"mera_fp32.hswish", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::HSwishFp>("HSwish", kType, GetShapeNchw(ir), inputs[0])};
    }},
    {"mera_fp32.hswish_onnx", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::HSwishFp>("HSwish", kType, GetShapeNchw(ir), inputs[0])};
    }},
    {"mera_fp32.maxpool2d", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto *attr = AsChecked<MaxPool2DAttrs>(ir.GetRootCall()->attrs);
      const int pool_h = GetInt(attr->pool_size[0]);
      const int pool_w = GetInt(attr->pool_size[1]);
      auto strides = AttrToMeraStrides(attr->strides);
      auto pad = AttrToMeraPadding(attr->padding);
      const auto out_shape = GetShapeNchw(ir);

      if (ir.Traverse().HasCall(0) && ir.Traverse()[0].IsOp("nn.pad")) {
        // Has explicit padding
        const auto *pad_attr = AsChecked<PadAttrs>(ir.Traverse()[0].GetCall()->attrs);
        pad.top += GetInt(pad_attr->pad_width[1][0]);
        pad.bottom += GetInt(pad_attr->pad_width[1][1]);
        pad.left += GetInt(pad_attr->pad_width[2][0]);
        pad.right += GetInt(pad_attr->pad_width[2][1]);
      }

      // Check if we need to add extra padding (when ceil_mode=True)
      if (attr->ceil_mode) {
        auto in_type = GetShapeNchw(ir.GetRootCall()->args[0]->checked_type());
        const int i_h = in_type.shape[2];
        const int i_w = in_type.shape[3];
        int o_h = std::floor(static_cast<float>(i_h - pool_h + pad.top + pad.bottom) / strides.h) + 1;
        int o_w = std::floor(static_cast<float>(i_w - pool_w + pad.left + pad.right) / strides.w) + 1;
        // Sanity check
        CHECK_GE(out_shape.shape[2], o_h);
        CHECK_GE(out_shape.shape[3], o_w);

        // Add extra pad to right and bottom
        pad.bottom += (out_shape.shape[2] - o_h);
        pad.right += (out_shape.shape[3] - o_w);
      }
      return TensorVec_t{graph_.Add<mera::ir::MaxPool2d>("MaxPool2d", kType, out_shape, inputs[0],
        pool_h, pool_w, strides, pad)};
    }},
    {"mera_fp32.res_add", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 2);
      return TensorVec_t{graph_.Add<mera::ir::AddOp>("Add", kType, GetShapeNchwMultiDim(ir), inputs[0], inputs[1])};
    }},
    {"mera_fp32.res_add_constant", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto data = ir.Traverse().CompileConstant(0);
      return TensorVec_t{graph_.Add<mera::ir::AddOp>("Add", kType, GetShapeNchwMultiDim(ir), data, inputs[0])};
    }},
    {"mera_fp32.avg_pool2d", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::AvgPooling2d>("AvgPooling2d", kType, GetShapeNchw(ir), inputs[0])};
    }},
    {"mera_fp32.avg_pool2d_onnx", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::AvgPooling2d>("AvgPooling2d", kType, GetShapeNchw(ir), inputs[0])};
    }},
    {"mera_fp32.concatenate", [&](const auto &inputs, auto &ir) {
      CHECK_GT(inputs.size(), 1);
      int axis = GetInt(AsChecked<ConcatenateAttrs>(ir.GetRootCall()->attrs)->axis);
      return TensorVec_t{graph_.Add<mera::ir::Concatenate>("Concatenate", kType, GetShapeNchw(ir), inputs, axis)};
    }},
    {"mera_fp32.upsampling", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto attrs = AsChecked<Resize2DAttrs>(ir.GetRootCall()->attrs);
      return TensorVec_t{graph_.Add<mera::ir::UpsamplingFp>("Upsampling", kType, GetShapeNchw(ir), inputs[0],
        attrs->method, attrs->coordinate_transformation_mode)};
    }},
    {"mera_fp32.upsampling_concat", [&](const auto &inputs, auto &ir) {
      CHECK_GT(inputs.size(), 1);
      const auto ups_attrs = AsChecked<Resize2DAttrs>(ir.GetRootCall()->attrs);
      int axis = GetInt(AsChecked<ConcatenateAttrs>(ir.Traverse().Get(0).GetCall()->attrs)->axis);

      auto out_shape = GetShapeNchw(ir);
      int N = out_shape.shape[0];
      int H = out_shape.shape[2];
      int W = out_shape.shape[3];
      std::vector<mera::ir::Tensor> cat_inputs;
      for (const auto &input : inputs) {
        int C = input.shape.shape[1];
        mera::ir::Shape ups_out_shape = mera::ir::Shape{{N, C, H, W}, mera::ir::layout::NCHW};

        cat_inputs.emplace_back(graph_.Add<mera::ir::UpsamplingFp>("Upsampling", kType, ups_out_shape, input,
          ups_attrs->method, ups_attrs->coordinate_transformation_mode));
      }

      return TensorVec_t{graph_.Add<mera::ir::Concatenate>("Concatenate", kType, out_shape, cat_inputs, axis)};
    }},
    {"mera_fp32.hardtanh", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto *attr = AsChecked<ClipAttrs>(ir.GetRootCall()->attrs);
      float clip_min = attr->a_min;
      float clip_max = attr->a_max;
      return TensorVec_t{graph_.Add<mera::ir::HardTanh>("HardTanh", kType, GetShapeNchw(ir), inputs[0], clip_min, clip_max)};
    }},
    {"mera_fp32.gelu", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::GELU>("GELU", kType, GetShape(ir), inputs[0])};
    }},
    {"mera_fp32.sigmoid", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      return TensorVec_t{graph_.Add<mera::ir::Sigmoid>("Sigmoid", kType, GetShape(ir), inputs[0])};
    }},
    {"mera_fp32.layer_norm", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);

      mera::ir::Tensor bias, weight;
      const auto out_shape = GetShape(ir);
      bool has_bias = true;
      auto irt = ir.Traverse();
      if (ir.Traverse().MoveIf("cast").IsOp("add")) {
        bias = ir.Traverse().MoveIf("cast").CompileConstant(1, FLATTEN_W);
        weight = irt.MoveIf("cast")[0].CompileConstant(1, FLATTEN_W);
      } else {
        // Disabled bias.
        has_bias = false;
        bias = graph_.AddFloatVec(std::vector<float>(out_shape.shape[2], 0.0), mera::ir::layout::W);
        weight = irt.MoveIf("cast").CompileConstant(1, FLATTEN_W);
      }

      return TensorVec_t{graph_.Add<mera::ir::LayerNorm>("LayerNorm", kType, out_shape, inputs[0], weight, bias, has_bias)};
    }},
    {"mera_fp32.matmul", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      mera::ir::Tensor data;
      if (ir.Traverse().IsOp("nn.dense")) {
        data = ir.Traverse().CompileConstant(1);
      } else {
        data = ir.Traverse()[0].CompileConstant(1);
      }
      CHECK_EQ(data.shape.rank, 2) << "Matmul data has to be a 2D tensor.";

      return TensorVec_t{graph_.Add<mera::ir::MatMul>("MatMul", kType, GetShape(ir), inputs[0], data)};
    }},
    {"mera_fp32.attention", [&](const auto &inputs, auto &ir) {
      return EmitAttentionNode(inputs, ir, graph_);
    }},
    {"mera_fp32.attention_sliced", [&](const auto &inputs, auto &ir) {
      return EmitAttentionNode(inputs, ir, graph_);
    }},
    {"mera_fp32.attention_const_query", [&](const auto &inputs, auto &ir) {
      constexpr static bool kConstQuery = true;
      return EmitAttentionNode(inputs, ir, graph_, kConstQuery);
    }},
    {"mera_fp32.masked_attention", [&](const auto &inputs, auto &ir) {
      constexpr static bool kHasMask = true;
      return EmitAttentionNode(inputs, ir, graph_, false, kHasMask);
    }},
    {"mera_fp32.3d_add_broadcast", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto bias = ir.Traverse().CompileConstant(0);
      return TensorVec_t{graph_.Add<mera::ir::BiasAdd>("BiasAdd", kType, GetShape(ir), inputs[0], bias)};
    }},
    {"mera_fp32.2d_add_broadcast", [&](const auto &inputs, auto &ir) {
      CHECK_EQ(inputs.size(), 1);
      const auto bias = ir.Traverse()[0].CompileConstant(1);
      return TensorVec_t{graph_.Add<mera::ir::BiasAdd>("BiasAdd", kType, GetShape(ir), inputs[0], bias)};
    }},
    {"mera_fp32.transpose_nop", [&](const auto &inputs, auto &ir) {
      // This is a captured NOP operation. Return pass-through
      return inputs;
    }},
    {"mera_fp32.cast_nop", [&](const auto &inputs, auto &ir) {
      // This is a captured NOP operation. Return pass-through
      return inputs;
    }},
  }) {};

} // namespace tvm::relay::contrib