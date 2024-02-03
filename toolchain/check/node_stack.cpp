// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/node_stack.h"

#include "llvm/ADT/STLExtras.h"

namespace Carbon::Check {

auto NodeStack::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  output << "NodeStack:\n";
  for (auto [i, entry] : llvm::enumerate(stack_)) {
    auto parse_node_kind = parse_tree_->node_kind(entry.parse_node);
    output << "\t" << i << ".\t" << parse_node_kind;
    if (parse_node_kind == Parse::NodeKind::BindingPattern) {
      output << " -> " << entry.id.As<SemIR::NameId>();
    } else {
      // TODO: Use the appropriate Id type for the parse node.
      if (entry.id.As<SemIR::InstId>().is_valid()) {
        output << " -> " << entry.id.As<SemIR::InstId>();
      }
    }
    output << "\n";
  }
}

}  // namespace Carbon::Check
