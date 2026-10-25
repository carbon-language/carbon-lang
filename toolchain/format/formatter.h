// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_

#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/format/style.h"
#include "toolchain/format/token_info.h"
#include "toolchain/format/whitespace_manager.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Format {

// Implements `Format()`; see `format.h`. Constructed and `Run()` once.
//
// Walks the token stream, buffering tokens into unwrapped lines and using
// parse-tree annotations for spacing, brace layout, and verbatim error
// recovery.
class Formatter {
 public:
  explicit Formatter(const Parse::Tree* tree, const Style& style);

  // Formats into the internal buffer and returns true if the input had no lex
  // or parse errors. Must be called once before `TakeOutput`.
  auto Run() -> bool;

  // Returns the full formatted text, leaving the formatter empty.
  auto TakeOutput() -> std::string { return std::move(output_); }

 private:
  // Returns the raw source slice between `token - 1` and `token` when both
  // tokens are real tokens belonging to the same outermost error subtree, or
  // `std::nullopt` when the gap before `token` should be formatted normally.
  auto VerbatimGapBefore(Lex::TokenIndex token) const
      -> std::optional<llvm::StringRef>;

  // Lays out the buffered unwrapped line (if any), records each token's leading
  // whitespace into `whitespace_`, and clears `current_line_`.
  auto FlushLine() -> void;

  // Returns the number of blank lines in `gap` clamped to
  // `[0, style_.max_empty_lines_to_keep]`.
  auto ClampBlankLines(llvm::StringRef gap) const -> int;

  // Returns true if `comment` starting at `start_byte` is attached as a
  // same-line trailing comment after `prev_end`.
  auto IsAttachedTrailingComment(llvm::StringRef source, int prev_end,
                                 int start_byte,
                                 Lex::CommentIndex comment) const -> bool;

  // Returns the number of blank lines to keep before `next_start_byte` (up to
  // `style_.max_empty_lines_to_keep`), or 0 at the start or end of a block.
  auto ComputeBlankLines(int next_start_byte, bool is_block_end) -> int;

  // Returns the leading newline count before the next content item: 0 before
  // the first item in the file, otherwise 1 plus any preserved blank line.
  auto LeadingNewlines(int next_start_byte, bool is_block_end) -> int {
    return (started_ ? 1 : 0) +
           ComputeBlankLines(next_start_byte, is_block_end);
  }

  // Returns the next non-recovery token at or after `token + 1`.
  auto NextRealToken(Lex::TokenIndex token) const -> Lex::TokenIndex;

  static auto NextToken(Lex::TokenIndex token) -> Lex::TokenIndex {
    return Lex::TokenIndex(token.index + 1);
  }

  const Parse::Tree* tree_;
  const Lex::TokenizedBuffer* tokens_;
  Style style_;

  WhitespaceManager whitespace_;
  std::string output_;

  TokenInfoStore token_infos_;
  llvm::SmallVector<Lex::TokenIndex> current_line_;

  std::optional<int> last_end_byte_;
  bool after_open_brace_ = false;
  bool started_ = false;
  int indent_ = 0;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
