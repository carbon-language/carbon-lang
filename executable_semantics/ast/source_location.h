// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_SOURCE_LOCATION_H_
#define EXECUTABLE_SEMANTICS_AST_SOURCE_LOCATION_H_

#include <string>

#include "common/ostream.h"

namespace Carbon {

struct SourceLocation {
  bool operator==(SourceLocation other) {
    // Pointer equality is often sufficient for filenames, but double-check the
    // string if needed.
    return (filename == other.filename ||
            strcmp(filename, other.filename) == 0) &&
           line_num == other.line_num;
  }

  void Print(llvm::raw_ostream& out) const {
    out << *filename << ":" << line_num;
  }
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // The filename should be arena-allocated to eliminate copies.
  const char* filename;
  int line_num;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_SOURCE_LOCATION_H_
