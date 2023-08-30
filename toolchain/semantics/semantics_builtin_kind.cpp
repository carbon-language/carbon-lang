// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_builtin_kind.h"

namespace Carbon::SemIR {

CARBON_DEFINE_ENUM_CLASS_NAMES(BuiltinKind) = {
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name) \
  CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/semantics/semantics_builtin_kind.def"
};

auto BuiltinKind::label() -> llvm::StringRef {
  static constexpr llvm::StringLiteral Labels[] = {
#define CARBON_SEMANTICS_BUILTIN_KIND(Name, Label) Label,
#include "toolchain/semantics/semantics_builtin_kind.def"
  };
  return Labels[AsInt()];
}

}  // namespace Carbon::SemIR
