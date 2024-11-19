// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/builtin_inst_kind.h"

namespace Carbon::SemIR {

CARBON_DEFINE_ENUM_CLASS_NAMES(BuiltinInstKind) = {
#define CARBON_SEM_IR_BUILTIN_INST_KIND(Name) \
  CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/sem_ir/inst_kind.def"
};

}  // namespace Carbon::SemIR
