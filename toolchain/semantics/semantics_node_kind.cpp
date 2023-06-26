// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

CARBON_DEFINE_ENUM_CLASS_NAMES(SemanticsNodeKind) = {
#define CARBON_SEMANTICS_NODE_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/semantics/semantics_node_kind.def"
};

auto SemanticsNodeKind::terminator_kind() const -> SemanticsTerminatorKind {
  static constexpr SemanticsTerminatorKind Table[] = {
#define CARBON_SEMANTICS_NODE_KIND(Name) SemanticsTerminatorKind::NotTerminator,
#define CARBON_SEMANTICS_NODE_KIND_WITH_TERMINATOR_KIND(Name, TerminatorKind) \
  SemanticsTerminatorKind::TerminatorKind,
#include "toolchain/semantics/semantics_node_kind.def"
  };
  return Table[AsInt()];
}

}  // namespace Carbon
