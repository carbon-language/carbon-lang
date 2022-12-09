// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

// An X-macro for defining the enumeration of SemanticIR builtins.
#define CARBON_SEMANTICS_BUILTIN_KINDS(X)                                     \
  /* Tracks expressions which are valid as types.                             \
   */                                                                         \
  X(TypeType)                                                                 \
                                                                              \
  /* Used when a SemanticNode has an invalid type, which should then be       \
   * ignored for future type checking.                                        \
   */                                                                         \
  X(InvalidType)                                                              \
                                                                              \
  /* The type of integers and integer literals, currently always i32.         \
   * Long-term we may not want it this way, but for now this is the approach. \
   */                                                                         \
  X(IntegerType)                                                              \
                                                                              \
  /* The type of reals and real literals, currently always f64. Long-term     \
   * we may not want it this way, but for now this is the approach.           \
   */                                                                         \
  X(RealType)                                                                 \
                                                                              \
  /* Keep invalid last, so that we can use values as array indices without    \
   * needing an invalid entry.                                                \
   */                                                                         \
  X(Invalid)

CARBON_ENUM_BASE(SemanticsBuiltinKindBase, CARBON_SEMANTICS_BUILTIN_KINDS)

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
