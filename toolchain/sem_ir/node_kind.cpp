// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/node_kind.h"

#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

CARBON_DEFINE_ENUM_CLASS_NAMES(NodeKind) = {
#define CARBON_SEM_IR_NODE_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/sem_ir/node_kind.def"
};

auto NodeKind::ir_name() const -> llvm::StringLiteral {
  return definition().ir_name();
}

auto NodeKind::value_kind() const -> NodeValueKind {
  static constexpr NodeValueKind Table[] = {
#define CARBON_SEM_IR_NODE_KIND(Name) \
  HasTypeId<SemIR::Name> ? NodeValueKind::Typed : NodeValueKind::None,
#include "toolchain/sem_ir/node_kind.def"
  };
  return Table[AsInt()];
}

auto NodeKind::terminator_kind() const -> TerminatorKind {
  return definition().terminator_kind();
}

auto NodeKind::definition() const -> const Definition& {
  static constexpr const Definition* Table[] = {
#define CARBON_SEM_IR_NODE_KIND(Name) &SemIR::Name::Kind,
#include "toolchain/sem_ir/node_kind.def"
  };
  return *Table[AsInt()];
}

}  // namespace Carbon::SemIR
