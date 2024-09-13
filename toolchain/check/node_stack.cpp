// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/node_stack.h"

#include "llvm/ADT/STLExtras.h"

namespace Carbon::Check {

auto NodeStack::PrintForStackDump(SemIR::Formatter& formatter, int indent,
                                  llvm::raw_ostream& output) const -> void {
  auto print_id = [&]<Id::Kind Kind>(Id id) {
    if constexpr (Kind == Id::Kind::None) {
      output << "no value\n";
    } else if constexpr (Kind == Id::Kind::Invalid) {
      CARBON_FATAL("Should not be in node stack");
    } else if constexpr (Kind == Id::KindFor<SemIR::InstId>()) {
      output << "\n";
      formatter.PrintInst(id.As<Id::KindFor<SemIR::InstId>()>(), indent + 4,
                          output);
    } else {
      output << id.As<Kind>() << "\n";
    }
  };

  output.indent(indent);
  output << "NodeStack:\n";
  for (auto [i, entry] : llvm::enumerate(stack_)) {
    auto node_kind = parse_tree_->node_kind(entry.node_id);
    output.indent(indent + 2);
    output << i << ". " << node_kind << ": ";
    switch (node_kind) {
#define CARBON_PARSE_NODE_KIND(Kind)                                        \
  case Parse::NodeKind::Kind:                                               \
    print_id.operator()<NodeKindToIdKind(Parse::NodeKind::Kind)>(entry.id); \
    break;
#include "toolchain/parse/node_kind.def"
    }
  }
}

}  // namespace Carbon::Check
