/*
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
 * \file src/relay/transforms/merge_compiler_regions.cc
 *
 * \brief After operators have been annotated with the targets that support
 * them, this pass creates regions of the operators for each target. It
 * is guaranteed that the regions will have a topological ordering so that
 * no data dependency issues exist.
 *
 * This pass only introduces annotations to indicate the regions.
 * partition_graph must subsequently be called to lift these regions out
 * as external functions.
 */

#include <tvm/ir/error.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../analysis/annotated_region_set.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {
namespace merge_compiler_region {

class RegionMerger : public MixedModeVisitor {
 public:
  explicit RegionMerger(AnnotatedRegionSet regions) : regions_(regions) {}

  void VisitExpr_(const CallNode* call) final {
    if (call->op == CompilerEndOp()) {
      auto region = regions_->GetRegion(GetRef<Call>(call));

      // Skip this region if it has been merged to the other region.
      if (merged_regions_.find(region->GetID()) != merged_regions_.end()) {
        return;
      }

      // Check the region target.
      auto compiler_attrs = call->attrs.as<CompilerAttrs>();
      ICHECK_EQ(region->GetTarget(), compiler_attrs->compiler);

      // Visit the unmerged parent regions.
      for (const auto& arg : region->GetInputs()) {
        // Region inputs must be begin annotation, and the region of
        // the begin annotation's argument is the parent region.
        auto begin = Downcast<Call>(arg);
        ICHECK_EQ(begin->op, CompilerBeginOp());
        auto parent_region = regions_->GetRegion(begin->args[0]);

        // Skip this region if it has been merged.
        if (!parent_region.defined()) {
          continue;
        } else if (merged_regions_.find(parent_region->GetID()) == merged_regions_.end()) {
          VisitExpr(begin->args[0]);
        }
      }

      // Collect unmerged parent regions.
      std::unordered_set<AnnotatedRegion, ObjectPtrHash, ObjectPtrEqual> mergeable_regions;
      for (const auto& arg : region->GetInputs()) {
        auto begin = Downcast<Call>(arg);
        ICHECK_EQ(begin->op, CompilerBeginOp());
        auto parent_region = regions_->GetRegion(begin->args[0]);
        if (parent_region.defined()) {
          mergeable_regions.insert(parent_region);
        }
      }

      // Propogate all the parent restrictions to the current region.
      auto& region_restrictions = region_restrictions_[region->GetID()];
      for (const auto& parent_region : mergeable_regions) {
        auto parent_restrictions = region_restrictions_[parent_region->GetID()];
        region_restrictions.insert(parent_restrictions.begin(), parent_restrictions.end());
      }

      for (const auto& parent_region : mergeable_regions) {
        // Skip the parent region with a different target.
        if (parent_region->GetTarget() != compiler_attrs->compiler) {
          region_restrictions.insert(parent_region->GetID());
          continue;
        }

        // Skip the parent region if it is in the restriction set.
        if (region_restrictions.find(parent_region->GetID()) != region_restrictions.end()) {
          continue;
        }

        // Merge the parent region to the current one.
        regions_->MergeRegions(parent_region, region);

        // Replace the parent region ID with the current region for all
        // other regions' restriction sets.
        for (const auto& r : regions_) {
          auto& restrictions = region_restrictions_[r->GetID()];
          if (restrictions.find(parent_region->GetID()) != restrictions.end()) {
            restrictions.erase(parent_region->GetID());
            restrictions.insert(region->GetID());
          }
        }
      }
      merged_regions_.insert(region->GetID());
    }
  }

 private:
  AnnotatedRegionSet regions_;
  std::unordered_set<int> merged_regions_;
  std::unordered_map<int, std::unordered_set<int>> region_restrictions_;
};

class MergeAnnotations : public ExprRewriter {
 public:
  explicit MergeAnnotations(AnnotatedRegionSet regions) : regions_(regions) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    // Merge annotations which are now internal to a region.
    // This happens if we see a compiler begin next to a
    // compiler end and they're both in the same region.
    if (call->op == CompilerBeginOp() && call->args[0]->IsInstance<CallNode>()) {
      auto arg = Downcast<Call>(call->args[0]);
      if (arg->op == CompilerEndOp()) {
        auto region1 = regions_->GetRegion(GetRef<Call>(call));
        auto region2 = regions_->GetRegion(arg);
        if (region1 == region2) {
          auto post_arg = post.as<CallNode>()->args[0];
          return post_arg.as<CallNode>()->args[0];
        }
      }
    }
    return post;
  }

 private:
  AnnotatedRegionSet regions_;
};

class NetworkExtractor : public ExprVisitor {
 public:
  explicit NetworkExtractor() {}

  Array<Expr> Extract(const Expr& expr) {
    networks_.clear();
    const auto& expr_body = expr.as<FunctionNode>()->body;
    // The last node in expr_body is always compiler_end annotation
    if (expr_body.as<CallNode>()->args[0].as<CallNode>()
        && expr_body.as<CallNode>()->args[0].as<CallNode>()->op == Op::Get("annotation.tuple_multi_networks")) {
      // Multiple networks
      VisitExpr(expr);
    } else {
      // A single network
      networks_.push_back(expr_body);
    }
    return networks_;
  }

 private:
  void VisitExpr_(const TupleNode* tuple) final {
    for (const auto& field : tuple->fields) {
      networks_.push_back(field);
    }
  }

  Array<Expr> networks_;
};

class TopoSorter {
 public:
  TopoSorter(const std::map<AnnotatedRegion, std::vector<AnnotatedRegion> >& adjlist) {
    adjlist_ = adjlist;
    visited_.clear();
    order_.clear();
  }

  void dfs(const AnnotatedRegion& src) {
    visited_[src] = true;
    for (const auto& dst : adjlist_[src]) {
      if (!visited_[dst])
        dfs(dst);
    }
    order_.push_front(src);
  }

  std::list<AnnotatedRegion> toposort() {
    for (const auto& x : adjlist_) {
      const AnnotatedRegion& src = x.first;
      if (!visited_[src]) {
        dfs(src);
      }
    }
    return order_;
  }

 private:
  std::map<AnnotatedRegion, std::vector<AnnotatedRegion> > adjlist_;
  std::map<AnnotatedRegion, bool> visited_;
  std::list<AnnotatedRegion> order_;
};

std::vector<AnnotatedRegion> toposort_subgraphs(const AnnotatedRegionSet& region_set) {
  std::vector<AnnotatedRegion> subgraphs;
  for (const auto& region : region_set) subgraphs.push_back(region);

  // Get the children of each subgraph
  std::map<AnnotatedRegion, std::vector<AnnotatedRegion> > children;
  for (const auto& src : subgraphs) {
    for (const auto& dst : subgraphs) {
      for (const auto& src_in : src->GetInputs()) {
        bool found = false;
        if (src_in.as<CallNode>()) {
          const auto& dst_outs = dst->GetOutputs();
          found = (std::find(dst_outs.begin(), dst_outs.end(), src_in.as<CallNode>()->args[0]) != dst_outs.end());
        }
        if (found) {
          children[src].push_back(dst);
          break;
        }
      }
    }
  }

  // Topological sort the subgraphs
  TopoSorter sorter(children);
  std::list<AnnotatedRegion> sorted_subgraphs = sorter.toposort();

  // Return the non-default subgraphs
  std::vector<AnnotatedRegion> outs;
  for (const auto& subgraph : sorted_subgraphs) {
    if (subgraph->GetTarget() != "default") {
      outs.push_back(subgraph);
    }
  }
  return outs;
}

std::vector<std::vector<Expr>> merge_region_sets(const Array<AnnotatedRegionSet>& region_sets) {
  int num_networks = region_sets.size();

  // Get the non-default subgraphs in each region_set and sort them
  std::vector<AnnotatedRegion> subgraphs[num_networks];
  int subgraph_counts[num_networks];
  for (int i = 0; i < num_networks; i++) {
    subgraphs[i] = toposort_subgraphs(region_sets[i]);
    subgraph_counts[i] = subgraphs[i].size();
  }

  // Find the region_set that has the min number of non-default subgraphs
  int id = 0;
  for (int i = 1; i < num_networks; i++) {
    if (subgraph_counts[i] < subgraph_counts[id]) {
      id = i;
    }
  }
  int min_subgraph_count = subgraph_counts[id];

  // Merge region_sets
  std::vector<std::vector<Expr>> to_be_merged;
  for (int c = 0; c < min_subgraph_count; c++) {
    std::vector<Expr> exprs;
    for (int i = 0; i < num_networks; i++) {
      const auto& region = subgraphs[i][c];
      for (const auto& expr : region->GetOutputs()) {
        exprs.push_back(expr);
      }
    }
    to_be_merged.push_back(exprs);
  }

  return to_be_merged;
}

static const PackedFunc* make_begin_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_begin");

static const PackedFunc* make_end_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_end");

class TupleInserter : public ExprRewriter {
 public:
  explicit TupleInserter(std::vector<Expr> exprs) : exprs_(exprs) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (call->op == CompilerEndOp()) {
      for (size_t i = 0; i < exprs_.size(); i++) {
        if (call == exprs_[i].as<CallNode>()) {
          // Create the tuple node if it has not been created
          if (!tuple_.as<TupleNode>()) {
            Array<Expr> fields;
            for (const auto& expr : exprs_) {
              fields.push_back(expr.as<CallNode>()->args[0]);
            }
            tuple_ = Tuple(fields);
          }
          // Create TupleGetItem
          auto tuple_get_item = TupleGetItem(tuple_, i);
          std::string target = call->attrs.as<CompilerAttrs>()->compiler;
          auto out = (*make_end_op)(tuple_get_item, target);
          // Set a boundary at the split point so that RegionMerger works correctly
          // auto out_1 = (*make_begin_op)(out, "default");
          // auto out_2 = (*make_end_op)(out_1, "default");
          return out;
        }
      }
    }
    return post;
  }

 private:
  std::vector<Expr> exprs_;
  Expr tuple_;
};

class ExprsBeforeConcatenateCollector : public ExprVisitor {
 public:
  explicit ExprsBeforeConcatenateCollector() {}

  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> collect(const Expr& expr) {
    exprs_before_concatenate_.clear();
    VisitExpr(expr);
    return exprs_before_concatenate_;
  }

 private:
  void VisitExpr_(const CallNode* call) final {
    auto is_valid_concat = [](const CallNode *call) -> bool {
      if (call && call->args.size()) {
        const auto *ptr1 = call->args[0].as<CallNode>();
        if (ptr1 && ptr1->args.size()) {
          const auto *ptr2 = ptr1->args[0].as<CallNode>();
          if (ptr2 && ptr2->args.size()) {
            const auto *ptr3 = ptr2->args[0].as<TupleNode>();
            if (ptr3) {
              return true;
            }
          }
        }
      }
      return false;
    };
    if (call->op == Op::Get("concatenate") && is_valid_concat(call)) {
      const TupleNode* tuple = call->args[0].as<CallNode>()->args[0].as<CallNode>()->args[0].as<TupleNode>();
      for (const auto& field : tuple->fields) {
        exprs_before_concatenate_.insert(field.as<CallNode>()->args[0].as<CallNode>()->args[0]);
      }
    }
    ExprVisitor::VisitExpr_(call);
  }

  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> exprs_before_concatenate_;
};

Expr MergeCompilerRegions(const Expr& expr) {
  // Extract the networks
  NetworkExtractor extractor;
  Array<Expr> networks = extractor.Extract(expr);
  int num_networks = networks.size();

  // Merge regions within each network
  Array<AnnotatedRegionSet> region_sets;
  for (int i = 0; i < num_networks; i++) {
    Expr network = networks[i];
    AnnotatedRegionSet region_set = AnnotatedRegionSet::Create(network, CompilerBeginOp(), CompilerEndOp());
    RegionMerger merger(region_set);
    merger.VisitExpr(network);
    region_sets.push_back(region_set);
  }

  // Group the networks
  Expr out;
  if (num_networks > 1) {
    Array<Expr> fields;
    for (int i = 0; i < num_networks; i++) {
      fields.push_back(networks[i]);
    }
    out = Tuple(fields);
  } else {
    out = networks[0];
  }

  // Merge independent regions from different networks
  if (num_networks > 1) {
    std::vector<std::vector<Expr>> to_be_merged = merge_region_sets(region_sets);
    // remove duplicate exprs before concatenate
    ExprsBeforeConcatenateCollector collector;
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> exprs_before_concatenate = collector.collect(out);
    for (size_t i = 0; i < to_be_merged.size(); i++) {
      std::vector<Expr>& exprs = to_be_merged[i];
      exprs.erase(std::remove_if(exprs.begin(),
                                 exprs.end(),
                                 [&](Expr x){return exprs_before_concatenate.count(x.as<CallNode>()->args[0]) > 0;}),
                  exprs.end());
    }

    for (size_t i = 0; i < to_be_merged.size(); i++) {
      TupleInserter inserter(to_be_merged[i]);
      out = PostOrderRewrite(out, &inserter);
    }
  }

  // Remove annotations that are not in the region boundaries
  AnnotatedRegionSet region_set = AnnotatedRegionSet::Create(out, CompilerBeginOp(), CompilerEndOp());
  RegionMerger merger(region_set);
  merger.VisitExpr(out);
  MergeAnnotations merge_anno(region_set);
  out = PostOrderRewrite(out, &merge_anno);

  // Return the updated function
  auto func = GetRef<Function>(expr.as<FunctionNode>());
  return Function(func->params, out, func->ret_type, func->type_params, func->attrs);
}

}  // namespace merge_compiler_region

namespace transform {

Pass MergeCompilerRegions() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> part_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(merge_compiler_region::MergeCompilerRegions(f));
      };
  auto merged = CreateFunctionPass(part_func, 0, "MergeCompilerRegions", {});
  return Sequential({merged, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.MergeCompilerRegions")
    .set_body_typed(transform::MergeCompilerRegions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
