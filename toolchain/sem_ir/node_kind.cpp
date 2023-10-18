// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::SemIR {

CARBON_DEFINE_ENUM_CLASS_NAMES(NodeKind) = {
#define CARBON_SEM_IR_NODE_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/sem_ir/node_kind.def"
};

// Returns the name to use for this node kind in Semantics IR.
[[nodiscard]] auto NodeKind::ir_name() const -> llvm::StringRef {
  static constexpr llvm::StringRef Table[] = {
#define CARBON_SEM_IR_NODE_KIND_WITH_IR_NAME(Name, IR_Name) IR_Name,
#include "toolchain/sem_ir/node_kind.def"
  };
  return Table[AsInt()];
}

auto NodeKind::value_kind() const -> NodeValueKind {
  static constexpr NodeValueKind Table[] = {
#define CARBON_SEM_IR_NODE_KIND_WITH_VALUE_KIND(Name, Kind) NodeValueKind::Kind,
#include "toolchain/sem_ir/node_kind.def"
  };
  return Table[AsInt()];
}

auto NodeKind::terminator_kind() const -> TerminatorKind {
  static constexpr TerminatorKind Table[] = {
#define CARBON_SEM_IR_NODE_KIND_WITH_TERMINATOR_KIND(Name, Kind) \
  TerminatorKind::Kind,
#include "toolchain/sem_ir/node_kind.def"
  };
  return Table[AsInt()];
}

}  // namespace Carbon::SemIR
