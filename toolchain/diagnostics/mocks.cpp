// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

namespace Carbon::Diagnostics {

auto PrintTo(const Diagnostic& diagnostic, std::ostream* os) -> void {
  *os << "Diagnostic{";
  PrintTo(diagnostic.level, os);
  for (const auto& message : diagnostic.messages) {
    *os << ", {" << message.loc.filename << ":" << message.loc.line_number
        << ":" << message.loc.column_number << ", \"" << message.Format()
        << "}";
  }
  *os << "\"}";
}

auto PrintTo(Level level, std::ostream* os) -> void {
  switch (level) {
    case Level::LocationInfo:
      *os << "LocationInfo";
      break;
    case Level::Note:
      *os << "Note";
      break;
    case Level::Warning:
      *os << "Warning";
      break;
    case Level::Error:
      *os << "Error";
      break;
  }
}

}  // namespace Carbon::Diagnostics
