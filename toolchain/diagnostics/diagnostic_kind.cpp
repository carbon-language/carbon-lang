// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_kind.h"

namespace Carbon {

auto operator<<(llvm::raw_ostream& out, DiagnosticKind kind)
    -> llvm::raw_ostream& {
  static constexpr llvm::StringLiteral Names[] = {
#define CARBON_DIAGNOSTIC_KIND(DiagnosticName) #DiagnosticName,
#include "toolchain/diagnostics/diagnostic_registry.def"
  };
  out << Names[static_cast<int>(kind)];
  return out;
}

}  // namespace Carbon
