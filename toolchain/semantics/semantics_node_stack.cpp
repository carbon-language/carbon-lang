// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node_stack.h"

#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsNodeStack::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  output << "SemanticsNodeStack:\n";
  for (int i = 0; i < static_cast<int>(stack_.size()); ++i) {
    const auto& entry = stack_[i];
    auto parse_node_kind = parse_tree_->node_kind(entry.parse_node);
    output << "\t" << i << ".\t" << parse_node_kind;
    if (parse_node_kind == ParseNodeKind::PatternBinding) {
      output << " -> " << entry.name_id;
    } else {
      if (entry.node_id.is_valid()) {
        output << " -> " << entry.node_id;
      }
    }
    output << "\n";
  }
}

}  // namespace Carbon
