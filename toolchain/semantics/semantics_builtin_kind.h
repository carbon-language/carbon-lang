// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_

#include <cstdint>

#include "common/ostream.h"

namespace Carbon {

#define CARBON_ENUM_BASE_NAME SemanticsBuiltinKindBase
#define CARBON_ENUM_DEF_PATH "toolchain/semantics/semantics_builtin_kind.def"
#include "toolchain/common/enum_base.def"

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

 protected:
  using SemanticsBuiltinKindBase::SemanticsBuiltinKindBase;
};

// We expect the builtin kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsBuiltinKind) == 1,
              "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
