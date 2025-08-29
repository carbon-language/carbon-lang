// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic.h"

#include <algorithm>
#include <cstdint>

#include "llvm/ADT/Sequence.h"

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
  if (!snippet.empty()) {
    llvm::StringRef snippet_ref = snippet;
    do {
      auto [snippet_line, rest] = snippet_ref.split('\n');
      out.indent(indent);
      out << snippet_line << "\n";
      snippet_ref = rest;
    } while (!snippet_ref.empty());
    return;
  }

  if (column_number == -1) {
    return;
  }
  // column_number is 1-based.
  const int caret_byte_offset = column_number - 1;

  out.indent(indent);

  int column = 0;
  int caret_column = 0;
  int underline_end_column = 0;

  int byte_offset = 0;
  for (char c : line) {
    // TODO: Handle tab characters.
    // TODO: Print Unicode characters directly, and use
    // llvm::sys::unicode::getColumnWidth to determine their width.
    if (std::isprint(static_cast<unsigned char>(c))) {
      out << c;
      ++column;
    } else {
      // TODO: Consider using ANSI colors to distinguish this from the program
      // text.
      int pos = out.tell();
      out << '<';
      llvm::write_hex(out, static_cast<unsigned char>(c),
                      llvm::HexPrintStyle::Upper, 2);
      out << '>';
      column += out.tell() - pos;
    }

    ++byte_offset;
    if (byte_offset <= caret_byte_offset) {
      caret_column = column;
    }
    if (byte_offset <= caret_byte_offset + length) {
      underline_end_column = column;
    }
  }

  out << "\n";

  out.indent(indent + caret_column);
  out << "^";
  // TODO: Revisit this once we can reference multiple ranges in a single
  // diagnostic message.
  for (auto _ :
       llvm::seq(std::max(underline_end_column - caret_column - 1, 0))) {
    out << '~';
  }
  out << '\n';
}

}  // namespace Carbon::Diagnostics
