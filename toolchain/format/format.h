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

// Formats the file represented by `tree` to `out` according to `style`.
// Produces best-effort output even when lex or parse errors exist, and returns
// true if the input had no errors.
auto Format(const Parse::Tree& tree, llvm::raw_ostream& out,
            const Style& style = Style()) -> bool;

// Computes the minimal edits that format `tree` according to `style`,
// appending them to `replacements`. Rewritten tokens (such as reformatted
// embedded C++ snippets) fold into their surrounding gap edit. If `lines` is
// set, only edits affecting that 1-based inclusive line range are produced;
// when `lines` is `std::nullopt`, applying the result to the source yields the
// same text `Format` writes. Produces best-effort edits even when lex or parse
// errors exist, and returns true if the input had no errors.
auto FormatReplacements(const Parse::Tree& tree,
                        llvm::SmallVectorImpl<Replacement>& replacements,
                        std::optional<LineRange> lines = std::nullopt,
                        const Style& style = Style()) -> bool;

// Applies `replacements` (ordered by `offset` and non-overlapping, as produced
// by `FormatReplacements`) to `source`, returning the edited text.
auto ApplyReplacements(llvm::StringRef source,
                       llvm::ArrayRef<Replacement> replacements) -> std::string;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMAT_H_
