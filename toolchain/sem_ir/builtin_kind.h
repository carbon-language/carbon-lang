// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BUILTIN_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_BUILTIN_KIND_H_

#include <cstdint>

#include "common/enum_base.h"

namespace Carbon::SemIR {

CARBON_DEFINE_RAW_ENUM_CLASS(BuiltinKind, uint8_t) {
#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/sem_ir/builtin_kind.def"
};

class BuiltinKind : public CARBON_ENUM_BASE(BuiltinKind) {
 public:
#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/sem_ir/builtin_kind.def"

  auto label() -> llvm::StringRef;

  // The count of enum values excluding Invalid.
  //
  // Note that we *define* this as `constexpr` making it a true compile-time
  // constant.
  static const uint8_t ValidCount;

  // Support conversion to and from an int32_t for SemIR instruction storage.
  using EnumBase::AsInt;
  using EnumBase::FromInt;
};

#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(BuiltinKind, Name)
#include "toolchain/sem_ir/builtin_kind.def"

constexpr uint8_t BuiltinKind::ValidCount = Invalid.AsInt();

static_assert(
    BuiltinKind::ValidCount != 0,
    "The above `constexpr` definition of `ValidCount` makes it available in "
    "a `constexpr` context despite being declared as merely `const`. We use it "
    "in a static assert here to ensure that.");

// We expect the builtin kind to fit compactly into 8 bits.
static_assert(sizeof(BuiltinKind) == 1, "Kind objects include padding!");

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_BUILTIN_KIND_H_
