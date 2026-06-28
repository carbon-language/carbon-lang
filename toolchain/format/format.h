// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMAT_H_

#include <cstdint>
#include <optional>
#include <string>

#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/format/style.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Format {

// A minimal edit to the source text: replace the `length` bytes at byte
// `offset` with `text`. The replacements produced by `FormatReplacements` are
// ordered by `offset` and never overlap.
struct Replacement {
  int32_t offset;
  int32_t length;
  std::string text;
};

// An inclusive range of 1-based source line numbers, used to restrict
// formatting to part of a file.
struct LineRange {
  int first_line;
  int last_line;
};

// Formats file content to the out stream, driven by the parse tree (and the
// tokens it references), using `style` (canonical by default). Returns false
// if the input had lex or parse errors; best-effort formatted output is still
// produced, and the caller decides whether to use it (the driver does, and
// reflects errors in its exit code).
auto Format(const Parse::Tree& tree, llvm::raw_ostream& out,
            const Style& style = Style()) -> bool;

// Computes the edits that format the document, appending them to
// `replacements`. Because formatting only ever changes whitespace and comments
// -- never the tokens themselves -- each edit covers one changed run of
// whitespace/comments between two tokens; unchanged runs produce no edit.
// Applying the result to the source yields the same text `Format` would write.
//
// If `lines` is set, only edits touching that line range are produced (the rest
// of the file is left unchanged). `style` defaults to the canonical style. The
// return value matches `Format`: false if the input had errors, though
// best-effort edits are produced regardless.
auto FormatReplacements(const Parse::Tree& tree,
                        llvm::SmallVectorImpl<Replacement>& replacements,
                        std::optional<LineRange> lines = std::nullopt,
                        const Style& style = Style()) -> bool;

// Applies `replacements` (as produced by `FormatReplacements`: ordered by
// offset and non-overlapping) to `source`, returning the edited text.
auto ApplyReplacements(llvm::StringRef source,
                       llvm::ArrayRef<Replacement> replacements) -> std::string;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
