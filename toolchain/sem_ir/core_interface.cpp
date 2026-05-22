// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/core_interface.h"  // IWYU pragma: keep

#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::SemIR {
CARBON_DEFINE_ENUM_CLASS_NAMES(CoreInterface) {
#define CARBON_SEM_IR_CORE_INTERFACE_KIND(Name) \
  CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/sem_ir/core_interface_kind.def"
};
}  // namespace Carbon::SemIR
