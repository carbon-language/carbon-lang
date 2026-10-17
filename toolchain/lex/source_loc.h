// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_
#define CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_

#include "common/check.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Lex {

// The location of a lexer diagnostic: a range of the source buffer. An empty
// range names the point between two characters, where something missing would
// be written.
//
// A `const char*` converts implicitly to the one character it points at, and
// any other range has to be made by a factory function. Converting a bare
// `const char*` to `llvm::StringRef` scans to the next null, which in a source
// buffer is the end of the file.
class SourceLoc {
 public:
  // The character at `pos`, which must point into the source buffer.
  //
  // NOLINTNEXTLINE(google-explicit-constructor): A character is a location.
  SourceLoc(const char* pos) : text_(pos, 1) {}

  // The range `text` covers, which must point into the source buffer.
  static auto Range(llvm::StringRef text) -> SourceLoc {
    return SourceLoc(text);
  }

  // The range from `begin` up to `end`, both of which must point into the
  // source buffer.
  static auto Range(const char* begin, const char* end) -> SourceLoc {
    CARBON_CHECK(begin <= end, "A range's end can't come before its start.");
    return SourceLoc(llvm::StringRef(begin, end - begin));
  }

  // The point just before `pos`, which must point into the source buffer or
  // just past its end.
  static auto Before(const char* pos) -> SourceLoc { return Range(pos, pos); }

  // The source text this location covers.
  auto text() const -> llvm::StringRef { return text_; }

 private:
  explicit SourceLoc(llvm::StringRef text) : text_(text) {}

  llvm::StringRef text_;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_
