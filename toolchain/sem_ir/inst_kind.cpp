// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_kind.h"

#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

CARBON_DEFINE_ENUM_CLASS_NAMES(InstKind) = {
#define CARBON_SEM_IR_INST_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/sem_ir/inst_kind.def"
};

auto InstKind::ir_name() const -> llvm::StringLiteral {
  return definition().ir_name();
}

auto InstKind::value_kind() const -> InstValueKind {
  static constexpr InstValueKind Table[] = {
#define CARBON_SEM_IR_INST_KIND(Name) \
  HasTypeId<SemIR::Name> ? InstValueKind::Typed : InstValueKind::None,
#include "toolchain/sem_ir/inst_kind.def"
  };
  return Table[AsInt()];
}

auto InstKind::terminator_kind() const -> TerminatorKind {
  return definition().terminator_kind();
}

auto InstKind::definition() const -> const Definition& {
  static constexpr const Definition* Table[] = {
#define CARBON_SEM_IR_INST_KIND(Name) &SemIR::Name::Kind,
#include "toolchain/sem_ir/inst_kind.def"
  };
  return *Table[AsInt()];
}

}  // namespace Carbon::SemIR
