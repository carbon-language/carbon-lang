// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

namespace Carbon {

void PrintTo(const Diagnostic& diagnostic, std::ostream* os) {
  *os << "Diagnostic{" << diagnostic.main.kind << ", ";
  PrintTo(diagnostic.level, os);
  *os << ", " << diagnostic.main.location.file_name << ":"
      << diagnostic.main.location.line_number << ":"
      << diagnostic.main.location.column_number << ", \""
      << diagnostic.main.format_fn(diagnostic.main) << "\"}";
}

void PrintTo(DiagnosticLevel level, std::ostream* os) {
  switch (level) {
    case DiagnosticLevel::Note:
      *os << "Note";
      break;
    case DiagnosticLevel::Warning:
      *os << "Warning";
      break;
    case DiagnosticLevel::Error:
      *os << "Error";
      break;
  }
}

}  // namespace Carbon
