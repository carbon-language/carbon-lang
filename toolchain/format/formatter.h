// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/format.h"
#include "toolchain/format/style.h"
#include "toolchain/format/token_info.h"
#include "toolchain/format/whitespace_manager.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Format {

// Implements Format(); see format.h. It's intended to be constructed and
// `Run()` once, then destructed.
//
// This walks the token stream, using the parse tree to drive spacing decisions.
// Tokens are buffered into the current unwrapped line; each line is then laid
// out as a unit, wrapped across physical lines by the solver in
// `line_wrapper.h` when it exceeds the column limit.
//
// `Run()` records each token's leading whitespace into a `WhitespaceManager`,
// then generates the full formatted text from it (running the trailing-comment
// alignment pass) and records where each token landed. The result is then
// available either as the whole text (`TakeOutput`) or as a minimal set of
// edits against the original source (`ComputeReplacements`).
//
// TODO: Drive indentation and line structure from the parse tree rather than
// from brace nesting.
class Formatter {
 public:
  explicit Formatter(const Parse::Tree* tree, const Style& style);

  // Formats into the internal buffer; see class comments. Must be called once
  // before `TakeOutput` or `ComputeReplacements`.
  auto Run() -> bool;

  // Returns the full formatted text, leaving the formatter empty.
  auto TakeOutput() -> std::string { return std::move(output_); }

  // Returns the edits that turn the original source into the formatted text:
  // one per changed run of whitespace/comments between tokens (see `format.h`).
  // If `lines` is set, only edits touching that 1-based inclusive line range
  // are returned.
  auto ComputeReplacements(std::optional<LineRange> lines) const
      -> llvm::SmallVector<Replacement>;

 private:
  // Returns the source byte ranges affected when formatting `lines`: the
  // requested lines, expanded (to a fixed point) over two layout couplings.
  // A partially affected unwrapped line is wholly affected, since it re-wraps
  // as a unit, and a brace whose matching brace is affected is affected too, so
  // range formatting fixes a dangling brace. See toolchain/docs/format.md.
  auto AffectedByteRanges(LineRange lines) const
      -> llvm::SmallVector<std::pair<int32_t, int32_t>>;

  // Lays out the buffered line (if any): decides its line breaks, then records
  // each token's leading whitespace into the whitespace manager, prefixed by
  // any blank line the source had before it. Clears the line buffer.
  auto FlushLine() -> void;

  // Returns the number of blank lines to keep before the content starting at
  // `next_start_byte`: one if the source had one or more there, else zero,
  // except none at the start or end of a block. Capped at the style maximum.
  auto ComputeBlankLines(int next_start_byte, bool is_block_end) -> int;

  // The leading newline count for the next content: a blank-line allowance plus
  // the single break that ends the previous line, or zero before the first
  // content of the file.
  auto LeadingNewlines(int next_start_byte, bool is_block_end) -> int {
    return (started_ ? 1 : 0) +
           ComputeBlankLines(next_start_byte, is_block_end);
  }

  // Returns the next token index.
  static auto NextToken(Lex::TokenIndex token) -> Lex::TokenIndex {
    return *(Lex::TokenIterator(token) + 1);
  }

  // The parse tree being formatted.
  const Parse::Tree* tree_;

  // The tokens being formatted, referenced by the parse tree.
  const Lex::TokenizedBuffer* tokens_;

  // The style controlling layout knobs and penalties.
  Style style_;

  // Collects each token's whitespace and generates the formatted text.
  WhitespaceManager whitespace_;

  // The formatted text, populated when `Run()` finishes.
  std::string output_;

  // The source-line extent (first and last 1-based line) of each flushed
  // unwrapped line, in order; used by `AffectedByteRanges` to expand a range to
  // whole unwrapped lines.
  llvm::SmallVector<std::pair<int, int>> unwrapped_line_extents_;

  // Where each emitted token landed in the source and the output, in order.
  llvm::SmallVector<TokenSpan> token_map_;

  // The per-token formatting information (role, width, and the
  // operator-precedence break and alignment data), indexed by token and
  // derived from the parse tree. See `TokenInfo`.
  TokenInfoStore token_infos_;

  // Tokens buffered for the current physical line, not yet rendered.
  llvm::SmallVector<Lex::TokenIndex> current_line_;

  // The source byte offset just past the last emitted token or comment, used to
  // find blank lines in the original source. Empty before any output.
  std::optional<int> last_end_byte_;

  // Whether the last rendered line ended with an opening `{`, so a blank line
  // at the start of a block is dropped.
  bool after_open_brace_ = false;

  // Whether any content has been recorded yet, so the first line gets no
  // leading newline.
  bool started_ = false;

  // The current code indent level, in spaces, added to new lines.
  int indent_ = 0;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
