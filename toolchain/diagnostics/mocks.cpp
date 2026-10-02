// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/mocks.h"

#include <ostream>

namespace Carbon::Diagnostics {

auto PrintTo(const Diagnostic& diagnostic, std::ostream* os) -> void {
  *os << "Diagnostic{";
  PrintTo(diagnostic.level, os);
  const Loc& loc = diagnostic.message.loc;
  *os << ", {" << loc.filename << ":" << loc.line_number << ":"
      << loc.column_number << ", \"" << diagnostic.message.Format() << "\"}";
  for (const auto& context : diagnostic.contexts) {
    *os << ", {Context " << context.loc.filename << ":"
        << context.loc.line_number << ":" << context.loc.column_number << ", \""
        << context.Format() << "\"}";
  }
  for (const auto& label : diagnostic.labels) {
    *os << ", {";
    PrintTo(label.category, os);
    *os << " " << label.loc.filename << ":" << label.loc.line_number << ":"
        << label.loc.column_number << ", \"" << label.Format() << "\"}";
  }
  *os << "}";
}

auto PrintTo(Level level, std::ostream* os) -> void {
  switch (level) {
    case Level::Warning:
      *os << "Warning";
      break;
    case Level::Error:
      *os << "Error";
      break;
  }
}

auto PrintTo(LabelCategory category, std::ostream* os) -> void {
  switch (category) {
    case LabelCategory::Info:
      *os << "Info";
      break;
    case LabelCategory::Primary:
      *os << "Primary";
      break;
  }
}

}  // namespace Carbon::Diagnostics
