// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/node_stack.h"

#include <utility>

#include "llvm/ADT/STLExtras.h"

namespace Carbon::Check {

auto NodeStack::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  static_assert(static_cast<std::size_t>(Id::Kind::Invalid) ==
                Id::NumValidKinds);
  static_assert(static_cast<std::size_t>(Id::Kind::None) ==
                Id::NumValidKinds + 1);
  using PrintIdFn = auto(llvm::raw_ostream & output, Id id)->void;
  using PrintIdFnsType = std::array<PrintIdFn*, Id::NumValidKinds + 2>;
  static constexpr PrintIdFnsType PrintIdFns =
      []<std::size_t... Is>(std::index_sequence<Is...>) -> PrintIdFnsType {
    return {[](llvm::raw_ostream& output, Id id) {
      constexpr auto IdKind = static_cast<Id::Kind>(Is);
      if constexpr (IdKind == Id::Kind::None) {
        output << " -> no value";
      } else if constexpr (IdKind == Id::Kind::Invalid) {
        CARBON_FATAL() << "Should not be in node stack";
      } else {
        output << id.As<IdKind>();
      }
    }...};
  }(std::make_index_sequence<Id::NumValidKinds + 2>());

  output << "NodeStack:\n";
  for (auto [i, entry] : llvm::enumerate(stack_)) {
    auto parse_node_kind = parse_tree_->node_kind(entry.parse_node);
    output << "\t" << i << ".\t" << parse_node_kind;
    PrintIdFns[static_cast<std::size_t>(
        ParseNodeKindToIdKind(parse_node_kind))](output, entry.id);
    output << "\n";
  }
}

}  // namespace Carbon::Check
