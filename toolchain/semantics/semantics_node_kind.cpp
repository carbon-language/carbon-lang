// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node_kind.h"

#include "llvm/ADT/StringRef.h"

namespace Carbon {

auto SemanticsNodeKind::name() const -> llvm::StringRef {
  static constexpr llvm::StringLiteral Names[] = {
#define CARBON_SEMANTICS_NODE_KIND(Name, ...) #Name,
#include "toolchain/semantics/semantics_node_kind.def"
  };
  return Names[static_cast<int>(kind_)];
}

}  // namespace Carbon
