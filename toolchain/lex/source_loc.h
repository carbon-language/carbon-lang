// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_
#define CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_

#include "common/check.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Lex {

// The location of a lexer diagnostic: a range of the source buffer, one
// character long when it names a position.
//
// A `const char*` converts implicitly to a one-character position, and a range
// has to be constructed explicitly. Converting a bare `const char*` to
// `llvm::StringRef` scans to the next null, which in a source buffer is the
// end of the file.
class SourceLoc {
 public:
  // The position at `pos`, which must point into the source buffer.
  //
  // NOLINTNEXTLINE(google-explicit-constructor): A position is a location.
  SourceLoc(const char* pos) : text_(pos, 1) {}

  // The range `text` covers, which must point into the source buffer.
  explicit SourceLoc(llvm::StringRef text) : text_(text) {}

  // The range from `begin` up to `end`, both of which must point into the
  // source buffer.
  static auto Range(const char* begin, const char* end) -> SourceLoc {
    CARBON_CHECK(begin <= end, "A range's end can't come before its start.");
    return SourceLoc(llvm::StringRef(begin, end - begin));
  }

  // The source text this location covers.
  auto text() const -> llvm::StringRef { return text_; }

 private:
  llvm::StringRef text_;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_
