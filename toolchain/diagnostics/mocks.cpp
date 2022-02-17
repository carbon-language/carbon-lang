// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

namespace Carbon {

void PrintTo(const Diagnostic& diagnostic, std::ostream* os) {
  *os << "Diagnostic{"
      << (diagnostic.level == Diagnostic::Level::Error ? "Error" : "Warning")
      << ", " << diagnostic.location.file_name << ":"
      << diagnostic.location.line_number << ":"
      << diagnostic.location.column_number << ", "
      << diagnostic.short_name.str() << ", " << diagnostic.message << "}";
}

}  // namespace Carbon
