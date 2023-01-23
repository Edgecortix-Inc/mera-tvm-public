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

#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/nn.h>

namespace tvm {
namespace relay {

struct MeraQuantizedResult : public tvm::AttrsNode<MeraQuantizedResult> {
  Type out_type;
  TVM_DECLARE_ATTRS(MeraQuantizedResult, "relay.attrs.MeraQuantizedResult") {
    TVM_ATTR_FIELD(out_type);
  }
};

TVM_REGISTER_NODE_TYPE(MeraQuantizedResult);

bool MeraQuantizedResultRel(const Array<Type>& types, int num_inputs,
    const Attrs& attrs, const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 3) << "Expects three types, one for the input, one for the data and another for the output";
  const auto* param = attrs.as<MeraQuantizedResult>();
  reporter->Assign(types[2], param->out_type);
  return true;
}

RELAY_REGISTER_OP("mera_quantized_result")
    .set_num_inputs(2)
    .add_argument("input", "Tensor", "Input tensors.")
    .add_argument("mera_data", "Tensor", "MERA info.")
    .set_support_level(3)
    .add_type_rel("MeraQuantizedResult", MeraQuantizedResultRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

Expr MakeMeraQuantizedResult(Expr input, Expr mera_data, Type out_type) {
    auto attrs = make_object<MeraQuantizedResult>();
    attrs->out_type = out_type;
    static const Op& op = Op::Get("mera_quantized_result");
    return Call(op, {input, mera_data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.mera_quantized_result").set_body_typed(MakeMeraQuantizedResult);

} // namespace relay
} // namespace tvm
