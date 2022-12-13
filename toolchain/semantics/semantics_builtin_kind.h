// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_ENUM_BASE_1_OF_7(SemanticsBuiltinKindBase)
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) CARBON_ENUM_BASE_2_OF_7_ITER(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"
CARBON_ENUM_BASE_3_OF_7(SemanticsBuiltinKindBase)
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) CARBON_ENUM_BASE_4_OF_7_ITER(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"
CARBON_ENUM_BASE_5_OF_7(SemanticsBuiltinKindBase)
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) CARBON_ENUM_BASE_6_OF_7_ITER(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"
CARBON_ENUM_BASE_7_OF_7(SemanticsBuiltinKindBase)

class SemanticsBuiltinKind
    : public SemanticsBuiltinKindBase<SemanticsBuiltinKind> {
 public:
  // The count of enum values excluding Invalid.
  static constexpr uint8_t ValidCount =
      static_cast<uint8_t>(InternalEnum::Invalid);

  // Support conversion to and from an int32_t for SemanticNode storage.
  auto AsInt() -> int32_t { return static_cast<int32_t>(val_); }
  static auto FromInt(int32_t val) -> SemanticsBuiltinKind {
    return SemanticsBuiltinKind(static_cast<InternalEnum>(val));
  }

 private:
  using SemanticsBuiltinKindBase::SemanticsBuiltinKindBase;
};

// We expect the builtin kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsBuiltinKind) == 1,
              "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
