// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

namespace Carbon {

void PrintTo(const Diagnostic& diagnostic, std::ostream* os) {
  *os << "Diagnostic{"
      << (diagnostic.level == Diagnostic::Level::Error ? "Error" : "Warning")
      << ", " << diagnostic.short_name.str() << ", " << diagnostic.message
      << "}";
}

void PrintTo(const DiagnosticLocation& loc, std::ostream* os) {
  *os << "Loc{" << loc.file_name << ":" << loc.line_number << ":"
      << loc.column_number << "}";
}

}  // namespace Carbon
