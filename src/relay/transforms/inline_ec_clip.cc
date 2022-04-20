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
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <string>

using namespace tvm::runtime;

namespace tvm {
namespace relay {

class InlineExternClipTransformer : ExprMutator {
 public:
  explicit InlineExternClipTransformer(const std::string& clip_func_name, double clip_min,
                                       double clip_max)
      : clip_func_name_(clip_func_name), clip_min_(clip_min), clip_max_(clip_max) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    if (call_node->op->IsInstance<GlobalVarNode>()) {
      auto gv = Downcast<GlobalVar>(call_node->op);
      if (gv->name_hint == clip_func_name_) {
        auto args = call_node->args;
        auto clip_op = Op::Get("clip");
        auto attrs = make_object<ClipAttrs>();
        attrs->a_min = clip_min_;
        attrs->a_max = clip_max_;
        return Call(clip_op, args, Attrs(attrs), {});
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

  Function Inline(const Function& func) {
    return Function(func->params, VisitExpr(func->body), func->ret_type, func->type_params,
                    func->attrs);
  }

 private:
  std::string clip_func_name_;
  double clip_min_;
  double clip_max_;
};

bool IsExternClipFunction(Function func, double& clip_min, double& clip_max) {
  /* Determine if the function is of the form
def @ec_329(%ec_329_i0: Tensor[(1, 1000), int32]) -> Tensor[(1, 1000), int32] {
  %295 = fn (%FunctionVar_0_03: Tensor[(1, 1000), int32],  Composite="mera.clip") -> Tensor[(1, 1000),
int32] {
     clip(%FunctionVar_0_03, a_min=0f, a_max=255f)
  };
  %295(%ec_329_i0)
}
  and extract clip min and max
  */
  // TODO: Clean up Downcast mess, maybe using pattern matching?
  auto compiler_name = func->GetAttr<runtime::String>(attr::kCompiler);
  if (!compiler_name.defined() || compiler_name.value() != "mera") return false;

  class CountFunctions : public ExprVisitor {
   public:
    void VisitExpr_(const FunctionNode* func_node) final {
      ++count;
      ExprVisitor::VisitExpr_(func_node);
    }

    int count = 0;
  };

  auto body = func->body;
  auto inner_func_counter = CountFunctions();
  inner_func_counter.VisitExpr(body);
  auto num_funcs = inner_func_counter.count;

  if (num_funcs == 1 && body->IsInstance<CallNode>()) {
    auto call = Downcast<Call>(body);
    auto callee = call->op;
    if (callee->IsInstance<FunctionNode>()) {
      auto callee_func = Downcast<Function>(callee);
      auto composite_name = callee_func->GetAttr<runtime::String>(attr::kComposite);
      if (composite_name.defined() && composite_name.value() == "mera.clip") {
        auto clip_call = Downcast<Call>(callee_func->body);
        CHECK(clip_call.defined());
        auto clip_attrs = clip_call->attrs.as<ClipAttrs>();
        CHECK(clip_attrs);
        clip_min = clip_attrs->a_min;
        clip_max = clip_attrs->a_max;
        return true;
      }
    }
  }
  return false;
}

IRModule InlineExternClip(const IRModule& module) {
  for (const auto& kv : module->functions) {
    auto f = Downcast<Function>(kv.second);
    CHECK(f.defined()) << "Downcast to Function failed";
    double clip_min, clip_max;
    if (IsExternClipFunction(f, clip_min, clip_max)) {
      LOG(INFO) << "Found clip function: " << kv.first->name_hint;
      auto main_func = Downcast<Function>(module->Lookup("main"));
      auto inliner = InlineExternClipTransformer(kv.first->name_hint, clip_min, clip_max);
      module->Update(module->GetGlobalVar("main"), inliner.Inline(main_func));
    }
  }
  return module;
}

namespace transform {

Pass InlineExternClip() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::InlineExternClip(m); };
  return CreateModulePass(pass_func, 1, "InlineExternClip", {});
}

TVM_REGISTER_GLOBAL("relay._transform.InlineExternClip").set_body_typed(InlineExternClip);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
