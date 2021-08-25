// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_SOURCE_LOCATION_H_
#define EXECUTABLE_SEMANTICS_AST_SOURCE_LOCATION_H_

#include <string>

#include "common/ostream.h"
#include "executable_semantics/common/ptr.h"

namespace Carbon {

class SourceLocation {
 public:
  // The filename should be eternal or arena-allocated to eliminate copies.
  SourceLocation(const char* filename, int line_num)
      : filename(filename), line_num(line_num) {}
  SourceLocation(Ptr<const std::string> filename, int line_num)
      : filename(filename->c_str()), line_num(line_num) {}

  SourceLocation(const SourceLocation&) = default;
  SourceLocation(SourceLocation&&) = default;
  auto operator=(const SourceLocation&) -> SourceLocation& = default;
  auto operator=(SourceLocation&&) -> SourceLocation& = default;

  bool operator==(SourceLocation other) const {
    return strcmp(filename.Get(), other.filename.Get()) == 0 &&
           line_num == other.line_num;
  }

  void Print(llvm::raw_ostream& out) const {
    out << filename.Get() << ":" << line_num;
  }
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  Ptr<const char> filename;
  int line_num;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_SOURCE_LOCATION_H_
