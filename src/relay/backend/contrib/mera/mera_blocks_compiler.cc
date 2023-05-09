/*
 * Copyright 2023 EdgeCortix Inc
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
#include "mera_compiler.h"
#include "../../utils.h"

#include <mera/mdna_blocks.h>

namespace tvm {
namespace relay {
namespace contrib {

inline const CallNode *ToCall(auto &x) { return AsChecked<CallNode>(x); }

/**
 * Keeps traversing through the IR calls until it finds operator 'op_name'. Returns that call.
 */
const CallNode *SearchUp(const CallNode *call, const std::string &op_name) {
  if (AsChecked<OpNode>(call->op)->name != op_name) {
    return SearchUp(ToCall(call->args[0]), op_name);
  }
  return call;
}

/**
 * @brief Traverses through a captured Yolov5 post IR function and extracts the original model's resolution:
 * [img_h x img_w].
 *
 * @tparam Qtz: Whether the encoded image size is quantized or not.
 */
template<bool Qtz>
std::pair<int, int> GetYolov5ImgSize(const FunctionNode *block_ir) {
  const std::string concat_op_name = Qtz ? "qnn.concatenate" : "concatenate";
  const std::string mul_op_name = Qtz ? "qnn.mul" : "multiply";
  auto *main_cat = AsChecked<TupleNode>(SearchUp(ToCall(block_ir->body), concat_op_name)->args[0]);
  CHECK_EQ(main_cat->fields.size(), 3);
  // Use field 2: feature with the largest stride.
  auto *b2_cat = AsChecked<TupleNode>(SearchUp(ToCall(main_cat->fields[2]), concat_op_name)->args[0]);
  CHECK_EQ(b2_cat->fields.size(), 3);
  // Use field 0 ("xy"), and search for the first mul. It will be a mul by [1/img_w, 1/img_h]
  auto *mul_node = SearchUp(ToCall(b2_cat->fields[0]), mul_op_name);
  auto *img_size = AsChecked<ConstantNode>(mul_node->args[1]);
  CHECK_EQ(GetShape(img_size->checked_type()).size, 2);
  float img_h_inv, img_w_inv;
  int check_dim;
  if constexpr (Qtz) {
    CHECK(GetType(img_size->checked_type()) == mera::ir::DataType::Int8);
    auto *scl = AsChecked<ConstantNode>(mul_node->args[4]);
    auto *zp = AsChecked<ConstantNode>(mul_node->args[5]);
    CHECK_EQ(GetShape(scl->checked_type()).size, 1);
    CHECK_EQ(GetShape(zp->checked_type()).size, 1);
    CHECK(GetType(scl->checked_type()) == mera::ir::DataType::Float32);
    CHECK(GetType(zp->checked_type()) == mera::ir::DataType::Int32);

    const int8_t *img_size_ptr = static_cast<const int8_t*>(img_size->data->data);
    const int8_t img_w_inv_qtz = img_size_ptr[0];
    const int8_t img_h_inv_qtz = img_size_ptr[1];
    const float scl_val = *static_cast<const float*>(scl->data->data);
    const int32_t zp_val = *static_cast<const int32_t*>(zp->data->data);
    auto deqtz = [scl_val, zp_val](int8_t x) -> float { return scl_val * (static_cast<int32_t>(x) - zp_val); };
    // Dequantize inverse resolution.
    img_h_inv = deqtz(img_h_inv_qtz);
    img_w_inv = deqtz(img_w_inv_qtz);
    check_dim = 2;
  } else {
    CHECK(GetType(img_size->checked_type()) == mera::ir::DataType::Float32);
    const float *img_size_ptr = static_cast<const float*>(img_size->data->data);
    img_w_inv = img_size_ptr[0];
    img_h_inv = img_size_ptr[1];
    check_dim = 1; // fp32 post has different layout.
  }

  // compute its inverse (mul of 1/x) and round to the next multiple of strides.
  constexpr static int k_stride = 32;
  auto round_up = [](int32_t x, int32_t to) -> int32_t { return ((x + to - 1) / to) * to; };
  int img_h = round_up(1.0 / img_h_inv, k_stride);
  int img_w = round_up(1.0 / img_w_inv, k_stride);

  // Double check that computed resolutions are correct by verifying:
  // qnn_mul.shape[check_dim] == (img_w * img_h) / (stride^2)
  //LOG(INFO) << "img_h=" << img_h << " img_w=" << img_w;
  CHECK_EQ(GetShape(mul_node->checked_type()).shape[check_dim], (img_h * img_w) / (k_stride * k_stride));
  return std::make_pair(img_h, img_w);
}

std::pair<std::vector<float>, std::vector<int32_t>> GetYolov5QnnI8QtzParams(const FunctionNode *block_ir, const std::string &concat_op_name) {
  constexpr static int k_legs = 3;
  std::vector<float> scls(k_legs);
  std::vector<int32_t> zps(k_legs);
  auto *main_cat = AsChecked<TupleNode>(SearchUp(ToCall(block_ir->body), concat_op_name)->args[0]);
  CHECK_EQ(main_cat->fields.size(), k_legs);
  for (int l = 0; l < k_legs; ++l) {
    auto *b_cat = AsChecked<TupleNode>(SearchUp(ToCall(main_cat->fields[l]), concat_op_name)->args[0]);
    CHECK_EQ(b_cat->fields.size(), 3);
    // Take any branch up to the sigmoid and then look for the dequantize
    auto *sigmoid_call = SearchUp(ToCall(b_cat->fields[0]), "sigmoid");
    auto *deq_call = SearchUp(ToCall(sigmoid_call->args[0]), "qnn.dequantize");
    auto *scl = AsChecked<ConstantNode>(deq_call->args[1]);
    auto *zp = AsChecked<ConstantNode>(deq_call->args[2]);
    CHECK_EQ(GetShape(scl->checked_type()).size, 1);
    CHECK_EQ(GetShape(zp->checked_type()).size, 1);
    CHECK(GetType(scl->checked_type()) == mera::ir::DataType::Float32);
    CHECK(GetType(zp->checked_type()) == mera::ir::DataType::Int32);
    scls[l] = *static_cast<const float*>(scl->data->data);
    zps[l] = *static_cast<const int32_t*>(zp->data->data);
  }
  return std::make_pair(scls, zps);
}

std::vector<mera::ir::Shape> CheckYolov5PostShapes(const FunctionNode *block_ir) {
  std::vector<mera::ir::Shape> feat_shapes;
  for (auto arg : block_ir->params) {
    if (auto *var = arg.as<VarNode>()) {
      auto shape = GetShape(var->checked_type());
      // Ignore all inputs that are not the 4D tensors
      if (shape.rank == 4) {
        feat_shapes.emplace_back(shape);
      }
    }
  }
  // We should only have 3 shapes captured
  CHECK_EQ(feat_shapes.size(), 3) << "Incorrect number of captured input Tensors";
  // Get batch value from any of the feature shapes
  return feat_shapes;
}

/**
 * @brief Compiles a YOLOv5 post block whose source are int8 tensors.
 * @tparam Qtz: Whether the implementation of the post processing is quantized or not.
 */
template<bool Qtz>
std::vector<uint8_t> CompileYolov5I8Post(const FunctionNode *block_ir) {
  /* Get batch value */
  int batch = CheckYolov5PostShapes(block_ir)[0].shape[0];

  /* Compute img_h and img_w */
  auto [img_h, img_w] = GetYolov5ImgSize<Qtz>(block_ir);

  /* Extract dequantization parameters */
  auto [scales, zps] = GetYolov5QnnI8QtzParams(block_ir, Qtz ? "qnn.concatenate" : "concatenate");

  constexpr static int k_num_classes = 80; // hard-coded for now
  mera::blocks::Yolov5i8Post yolov5_post{batch, k_num_classes, img_h, img_w, scales, zps};
  //LOG(INFO) << "Created block " << yolov5_post;
  return yolov5_post.SaveParams();
}

/**
 * @brief Compiles a YOLOv5 post block whose source are float32 tensors.
 */
std::vector<uint8_t> CompileYolov5Post(const FunctionNode *block_ir) {
  /*
   * Get batch value and compute img_h and img_w
   * For this pattern we have the input tensors already reshaped as
   * [B, H / stride, W / stride, C]; so we can get them from the input shapes.
   */
  auto shapes = CheckYolov5PostShapes(block_ir);
  constexpr static int k_stride_small = 8;
  // Use the biggest dim[1], dim[2] and multiply by smallest stride to get img_h,img_w
  int dim_h = 0, dim_w = 0;
  for (const auto &in_shape : shapes) {
    dim_h = std::max(dim_h, in_shape.shape[1]);
    dim_w = std::max(dim_w, in_shape.shape[2]);
  }
  const int img_h = dim_h * k_stride_small;
  const int img_w = dim_w * k_stride_small;
  const int batch = shapes[0].shape[0];
  constexpr static int k_num_classes = 80; // hard-coded for now

  mera::blocks::Yolov5Post yolov5_post(batch, k_num_classes, img_h, img_w);
  //LOG(INFO) << "Created block " << yolov5_post;
  return yolov5_post.SaveParams();
}

std::pair<std::vector<uint8_t>, std::string> CompileBlock(const std::string &block_name, const FunctionNode *block_ir) {
  std::string block_id;
  std::vector<uint8_t> block_code;
  if (block_name == "mera_blocks.yolov5_fp32_post") {
    block_id = mera::blocks::Yolov5Post::GetBlockId();
    block_code = CompileYolov5Post(block_ir);
  } else if (block_name == "mera_blocks.yolov5_i8_post") {
    block_id = mera::blocks::Yolov5i8Post::GetBlockId();
    block_code = CompileYolov5I8Post<false>(block_ir);
  } else if (block_name == "mera_blocks.yolov5_qnn_i8_post") {
    block_id = mera::blocks::Yolov5i8Post::GetBlockId();
    block_code = CompileYolov5I8Post<true>(block_ir);
  } else {
    LOG(FATAL) << "Unsupported MERA block '" << block_name << "'";
  }
  return std::make_pair(std::move(block_code), block_id);
}

runtime::Module CompileModuleMeraBlocks(const ObjectRef &ref) {
  CHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  CHECK(func.defined()) << "Expected a Relay function";
  std::string func_name = backend::GetExtSymbol(func);
  auto *func_ptr = AsChecked<FunctionNode>(func);
  auto *func_call = AsChecked<CallNode>(func_ptr->body);
  auto *block_func = AsChecked<FunctionNode>(func_call->op);

  auto block_name = block_func->GetAttr<runtime::String>(attr::kComposite);
  CHECK(block_name.defined());
  std::string block_name_val = block_name.value();

  // TODO - Check captured MERA block region is a single block, we don't support multiple blocks in the same module.
  auto [block_code, block_id] = CompileBlock(block_name_val, block_func);
  DLTensor code;
  code.data = block_code.data();
  code.device = DLDevice{kDLCPU, 0};
  code.ndim = 1;
  code.dtype = DLDataType{kDLUInt, 8, 1};
  int64_t shape[] = {int64_t(block_code.size())};
  code.shape = shape;
  code.strides = nullptr;
  code.byte_offset = 0;

  const auto* pf = runtime::Registry::Get("runtime.module.mera_blocks_module_create");
  CHECK_NOTNULL(pf);

  return (*pf)(func_name, block_id, &code);
}

TVM_REGISTER_GLOBAL("relay.ext.mera_blocks").set_body_typed(CompileModuleMeraBlocks);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
