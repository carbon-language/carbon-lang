// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_kind.h"

namespace Carbon {

auto DiagnosticKind::name() const -> llvm::StringRef {
  static constexpr llvm::StringLiteral Names[] = {
#define DIAGNOSTIC_KIND(DiagnosticName) #DiagnosticName,
#include "toolchain/diagnostics/diagnostic_registry.def"
  };
  return Names[static_cast<int>(kind_value_)];
}

}  // namespace Carbon
