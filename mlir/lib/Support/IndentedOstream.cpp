//===- IndentedOstream.cpp - raw ostream wrapper to indent ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// raw_ostream subclass that keeps track of indentation for textual output
// where indentation helps readability.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/IndentedOstream.h"

using namespace mlir;

raw_indented_ostream &mlir::raw_indented_ostream::reindent(StringRef str) {
  StringRef remaining = str;
  // Find leading whitespace indent.
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    size_t indent = split.first.find_first_not_of(" \t");
    if (indent != StringRef::npos) {
      leadingWs = indent;
      break;
    }
    remaining = split.second;
  }
  // Print, skipping the empty lines.
  *this << remaining;
  leadingWs = 0;
  return *this;
}

void mlir::raw_indented_ostream::write_impl(const char *ptr, size_t size) {
  StringRef str(ptr, size);
  // Print out indented.
  auto print = [this](StringRef str) {
    if (atStartOfLine)
      os.indent(currentIndent) << str.substr(leadingWs);
    else
      os << str.substr(leadingWs);
  };

  while (!str.empty()) {
    size_t idx = str.find('\n');
    if (idx == StringRef::npos) {
      if (!str.substr(leadingWs).empty()) {
        print(str);
        atStartOfLine = false;
      }
      break;
    }

    auto split =
        std::make_pair(str.slice(0, idx), str.slice(idx + 1, StringRef::npos));
    // Print empty new line without spaces if line only has spaces.
    if (!split.first.ltrim().empty())
      print(split.first);
    os << '\n';
    atStartOfLine = true;
    str = split.second;
  }
}
