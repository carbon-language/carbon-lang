// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

namespace Carbon {

void PrintTo(const Diagnostic& diagnostic, std::ostream* os) {
  *os << "Diagnostic{" << diagnostic.message.kind << ", ";
  PrintTo(diagnostic.level, os);
  *os << ", " << diagnostic.message.location.file_name << ":"
      << diagnostic.message.location.line_number << ":"
      << diagnostic.message.location.column_number << ", \""
      << diagnostic.message.format_fn(diagnostic.message) << "\"}";
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
