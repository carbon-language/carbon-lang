// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/node_category.h"

#include "llvm/ADT/StringExtras.h"

namespace Carbon::Parse {

// Returns a string form of the node category for printing.
static auto NodeCategoryToString(NodeCategory::RawEnumType category)
    -> llvm::StringLiteral {
#define CARBON_NODE_CATEGORY_TO_STRING(Name) \
  case NodeCategory::Name:                   \
    return #Name;

  switch (category) {
    CARBON_NODE_CATEGORY(CARBON_NODE_CATEGORY_TO_STRING)
    CARBON_NODE_CATEGORY_TO_STRING(None)
  }

#undef CARBON_NODE_CATEGORY_TO_STRING
}

auto NodeCategory::Print(llvm::raw_ostream& out) const -> void {
  llvm::ListSeparator sep("|");
  auto value = value_;
  do {
    // The lowest set bit in the value, or 0 (`None`) if no bits are set.
    auto lowest_bit = static_cast<RawEnumType>(value & -value);
    out << sep << NodeCategoryToString(lowest_bit);
    value &= ~lowest_bit;
  } while (value);
}

}  // namespace Carbon::Parse
