// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic.h"

#include <algorithm>
#include <cstdint>

namespace Carbon::Diagnostics {

auto Loc::FormatLocation(llvm::raw_ostream& out) const -> void {
  if (filename.empty()) {
    return;
  }
  out << filename;
  if (line_number > 0) {
    out << ":" << line_number;
    if (column_number > 0) {
      out << ":" << column_number;
    }
  }
  out << ": ";
}

auto Loc::FormatSnippet(llvm::raw_ostream& out, int indent) const -> void {
  if (column_number == -1) {
    return;
  }

  // column_number is 1-based.
  int32_t column = column_number - 1;

  out.indent(indent);
  out << line << "\n";
  out.indent(indent + column);
  out << "^";
  // We want to ensure that we don't underline past the end of the line in
  // case of a multiline token.
  // TODO: Revisit this once we can reference multiple ranges on multiple
  // lines in a single diagnostic message.
  int underline_length =
      std::min(length, static_cast<int32_t>(line.size()) - column);
  for (int i = 1; i < underline_length; ++i) {
    out << '~';
  }
  out << '\n';
}

}  // namespace Carbon::Diagnostics
