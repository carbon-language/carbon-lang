// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

namespace Carbon {

void PrintTo(const Diagnostic& diagnostic, std::ostream* os) {
  // TODO: Get diagnostic.kind as a string.
  *os << "Diagnostic{";
  PrintTo(diagnostic.kind, os);
  *os << ", ";
  PrintTo(diagnostic.level, os);
  *os << ", " << diagnostic.location.file_name << ":"
      << diagnostic.location.line_number << ":"
      << diagnostic.location.column_number << ", \""
      << diagnostic.format_fn(diagnostic) << "\"}";
}

void PrintTo(DiagnosticKind level, std::ostream* os) {
  // TODO
  *os << static_cast<int>(level);
}

void PrintTo(DiagnosticLevel level, std::ostream* os) {
  *os << (level == DiagnosticLevel::Error ? "Error" : "Warning");
}

}  // namespace Carbon
