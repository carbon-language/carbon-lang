// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_DEFINE_RAW_ENUM_CLASS(SemanticsBuiltinKind, uint8_t) {
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"
};

class SemanticsBuiltinKind : public CARBON_ENUM_BASE(SemanticsBuiltinKind) {
 public:
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) \
  CARBON_ENUM_CONSTANT_DECLARATION(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"

  // The count of enum values excluding Invalid.
  //
  // Note that we *define* this as `constexpr` making it a true compile-time
  // constant, and so we name it accordingly and disable the lint error here.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static const uint8_t ValidCount;

  // Support conversion to and from an int32_t for SemanticNode storage.
  using EnumBase::AsInt;
  using EnumBase::FromInt;
};

#define CARBON_SEMANTICS_BUILTIN_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(SemanticsBuiltinKind, Name)
#include "toolchain/semantics/semantics_builtin_kind.def"

constexpr uint8_t SemanticsBuiltinKind::ValidCount = Invalid.AsInt();

static_assert(
    SemanticsBuiltinKind::ValidCount != 0,
    "The above `constexpr` definition of `ValidCount` makes it available in "
    "a `constexpr` context despite being declared as merely `const`. We use it "
    "in a static assert here to ensure that.");

// We expect the builtin kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsBuiltinKind) == 1,
              "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
