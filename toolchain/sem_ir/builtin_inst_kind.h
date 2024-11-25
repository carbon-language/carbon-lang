// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BUILTIN_INST_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_BUILTIN_INST_KIND_H_

#include <cstdint>

#include "common/enum_base.h"

namespace Carbon::SemIR {

CARBON_DEFINE_RAW_ENUM_CLASS(BuiltinInstKind, uint8_t) {
#define CARBON_SEM_IR_BUILTIN_INST_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/sem_ir/inst_kind.def"
};

class BuiltinInstKind : public CARBON_ENUM_BASE(BuiltinInstKind) {
 public:
#define CARBON_SEM_IR_BUILTIN_INST_KIND(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/sem_ir/inst_kind.def"

  // The count of enum values excluding Invalid.
  static constexpr uint8_t ValidCount = 0
#define CARBON_SEM_IR_BUILTIN_INST_KIND(Name) +1
#include "toolchain/sem_ir/inst_kind.def"
      ;

  // Support conversion to and from an int32_t for SemIR instruction storage.
  using EnumBase::AsInt;
  using EnumBase::FromInt;
};

#define CARBON_SEM_IR_BUILTIN_INST_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(BuiltinInstKind, Name)
#include "toolchain/sem_ir/inst_kind.def"

static_assert(
    BuiltinInstKind::ValidCount != 0,
    "The above `constexpr` definition of `ValidCount` makes it available in "
    "a `constexpr` context despite being declared as merely `const`. We use it "
    "in a static assert here to ensure that.");

// We expect the builtin kind to fit compactly into 8 bits.
static_assert(sizeof(BuiltinInstKind) == 1, "Kind objects include padding!");

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_BUILTIN_INST_KIND_H_
