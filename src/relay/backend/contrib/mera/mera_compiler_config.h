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
#ifndef TVM_RELAY_BACKEND_MERA_COMPILER_CONFIG_H
#define TVM_RELAY_BACKEND_MERA_COMPILER_CONFIG_H

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm::relay::contrib {

struct MeraCompilerConfigNode : public tvm::AttrsNode<MeraCompilerConfigNode> {
  String input_layout;
  String weight_layout;
  String mera_ccfg;
  String mera_arch;
  String mera_target;

  TVM_DECLARE_ATTRS(MeraCompilerConfigNode, "ext.attrs.MeraCompilerConfigNode") {
    TVM_ATTR_FIELD(input_layout).set_default("NHWC");
    TVM_ATTR_FIELD(weight_layout).set_default("OIHW");
    TVM_ATTR_FIELD(mera_ccfg).set_default("");
    TVM_ATTR_FIELD(mera_arch).set_default("");
    TVM_ATTR_FIELD(mera_target).set_default("");
  }
};

class MeraCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(MeraCompilerConfig, Attrs, MeraCompilerConfigNode);
};

struct MeraQtzCompilerConfigNode : public tvm::AttrsNode<MeraQtzCompilerConfigNode> {
  Map<String, Array<Array<String>>> q_params;

  TVM_DECLARE_ATTRS(MeraQtzCompilerConfigNode, "ext.attrs.MeraQtzCompilerConfigNode") {
    TVM_ATTR_FIELD(q_params);
  }
};

class MeraQtzCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(MeraQtzCompilerConfig, Attrs, MeraQtzCompilerConfigNode);
};

MeraCompilerConfig GetMeraCompilerConfig();

MeraQtzCompilerConfig GetMeraQtzCompilerConfig();

} // namespace tvm::relay::contrib

#endif // TVM_RELAY_BACKEND_MERA_COMPILER_CONFIG_H