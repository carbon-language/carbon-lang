// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_NODE_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_NODE_STACK_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// A rudimentary equivalent of `NodeStack` that holds pattern insts.
//
// This type exists to support an incremental transition to the new SemIR
// model for patterns and pattern matching. In the old model, we would
// directly emit pattern-matching insts while checking the pattern AST,
// whereas in the new model we emit non-executable pattern insts, and then
// traverse them to emit the pattern-matching insts.
//
// The two models need to use the node stack in different ways, and so
// during the transition, `NodeStack` continues to support the old model,
// while `PatternNodeStack` holds what `NodeStack` _will_ hold once we
// fully transition to the new model.
//
// Since this class is intended to be temporary, it provides only a subset
// of the `NodeStack` API, and that subset has minimal type/kind checking.
class PatternNodeStack {
 public:
  PatternNodeStack() = default;

  auto Push(Parse::NodeId node_id, SemIR::InstId inst_id) -> void {
    // FIXME CHECK that node_id is right?
    stack_.push_back({.node_id = node_id, .id = inst_id.index});
  }

  auto Push(Parse::NodeId node_id, SemIR::InstBlockId inst_block_id) -> void {
    // FIXME CHECK that node_id is right?
    stack_.push_back({.node_id = node_id, .id = inst_block_id.index});
  }

  // FIXME comments
  template <typename IdT>
  auto Pop(Parse::NodeId node_id) -> IdT {
    auto entry = stack_.pop_back_val();
    CARBON_CHECK(entry.node_id == node_id)
        << "Expected " << node_id << ", but found " << entry.node_id;
    return IdT(entry.id);
  }

  auto empty() -> bool { return stack_.empty(); }

 private:
  struct Entry {
    Parse::NodeId node_id;
    int32_t id;
  };

  llvm::SmallVector<Entry> stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_NODE_STACK_H_
