// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

namespace Carbon {

void PrintTo(const Diagnostic& diagnostic, std::ostream* os) {
  *os << "Diagnostic{";
  PrintTo(diagnostic.level, os);
  for (const auto& message : diagnostic.messages) {
    *os << ", {" << message.loc.filename << ":" << message.loc.line_number
        << ":" << message.loc.column_number << ", \""
        << message.format_fn(message) << "}";
  }
  *os << "\"}";
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
