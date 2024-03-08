// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/node_stack.h"

#include "llvm/ADT/STLExtras.h"

namespace Carbon::Check {

auto NodeStack::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  auto print_id = [&]<Id::Kind Kind>(Id id) {
    if constexpr (Kind == Id::Kind::None) {
      output << " -> no value";
    } else if constexpr (Kind == Id::Kind::Invalid) {
      CARBON_FATAL() << "Should not be in node stack";
    } else {
      output << " -> " << id.As<Kind>();
    }
  };

  output << "NodeStack:\n";
  for (auto [i, entry] : llvm::enumerate(stack_)) {
    auto node_kind = parse_tree_->node_kind(entry.node_id);
    output << "\t" << i << ".\t" << node_kind;
    switch (node_kind) {
#define CARBON_PARSE_NODE_KIND(Kind)                                        \
  case Parse::NodeKind::Kind:                                               \
    print_id.operator()<NodeKindToIdKind(Parse::NodeKind::Kind)>(entry.id); \
    break;
#include "toolchain/parse/node_kind.def"
    }
    output << "\n";
  }
}

}  // namespace Carbon::Check
