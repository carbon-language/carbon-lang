// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

namespace Internal {
enum class SemanticsBuiltinKindEnum : uint8_t {
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) CARBON_ENUM_BASE_LITERAL(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"
};
}  // namespace Internal

class SemanticsBuiltinKind
    : public EnumBase<SemanticsBuiltinKind,
                      Internal::SemanticsBuiltinKindEnum> {
 public:
  // The count of enum values excluding Invalid.
  static constexpr uint8_t ValidCount =
      static_cast<uint8_t>(InternalEnum::Invalid);

#define CARBON_SEMANTICS_BUILTIN_KIND(Name) \
  CARBON_ENUM_BASE_FACTORY(SemanticsBuiltinKind, Name)
#include "toolchain/semantics/semantics_builtin_kind.def"

  // Gets a friendly name for the token for logging or debugging.
  [[nodiscard]] inline auto name() const -> llvm::StringRef {
    static constexpr llvm::StringLiteral Names[] = {
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) CARBON_ENUM_BASE_STRING(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"
    };
    return Names[static_cast<int>(val_)];
  }

  // Support conversion to and from an int32_t for SemanticNode storage.
  auto AsInt() -> int32_t { return static_cast<int32_t>(val_); }
  static auto FromInt(int32_t val) -> SemanticsBuiltinKind {
    return SemanticsBuiltinKind(static_cast<InternalEnum>(val));
  }

 private:
  using EnumBase::EnumBase;
};

// We expect the builtin kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsBuiltinKind) == 1,
              "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
