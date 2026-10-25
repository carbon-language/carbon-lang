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

// Implements `Format()` and `FormatReplacements()`; see `format.h`. Constructed
// and `Run()` once.
//
// Walks the token stream, buffering tokens into unwrapped lines and using
// parse-tree annotations for spacing, line termination, and line wrapping.
// `Run()` records each token's and comment's leading whitespace into
// `WhitespaceManager`, runs trailing-comment alignment, and generates the
// formatted output and token anchor map so callers can take either the full
// text (`TakeOutput`) or minimal edits (`ComputeReplacements`).
//
// TODO: Drive unwrapped-line boundaries and indentation directly from the parse
// tree.
class Formatter {
 public:
  explicit Formatter(const Parse::Tree* tree, const Style& style);

  // Formats into the internal buffer and returns true if the input had no lex
  // or parse errors. Must be called once before `TakeOutput` or
  // `ComputeReplacements`.
  auto Run() -> bool;

  // Returns the full formatted text, leaving the formatter empty.
  auto TakeOutput() -> std::string { return std::move(output_); }

  // Returns minimal edits transforming the original source into `output_`. If
  // `lines` is set, only edits affecting that 1-based inclusive line range are
  // returned.
  auto ComputeReplacements(std::optional<LineRange> lines) const
      -> llvm::SmallVector<Replacement>;

 private:
  // Returns the sorted, non-overlapping source byte ranges affected when
  // formatting `lines`, expanding the requested lines to whole unwrapped lines,
  // aligned trailing-comment groups, and matching braces.
  auto AffectedByteRanges(LineRange lines) const
      -> llvm::SmallVector<std::pair<int32_t, int32_t>>;

  // Returns the raw source slice between `token - 1` and `token` when both
  // tokens are real tokens belonging to the same outermost error subtree, or
  // `std::nullopt` when the gap before `token` should be formatted normally.
  auto VerbatimGapBefore(Lex::TokenIndex token) const
      -> std::optional<llvm::StringRef>;

  // Lays out the buffered unwrapped line (if any), records each token's and
  // embedded comment's leading whitespace into `whitespace_`, and clears
  // `current_line_`.
  auto FlushLine() -> void;

  // Appends `token` to `current_line_`, recording any blank line following an
  // immediately preceding mid-statement comment.
  auto PushToken(Lex::TokenIndex token) -> void;

  // Returns true when `current_line_` ends before `token`, so any comments
  // preceding `token` belong after `current_line_` rather than inside it.
  auto ShouldFlushBeforeComments(Lex::TokenIndex token) const -> bool;

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

  // A trailing or full-line comment encountered inside `current_line_` before
  // `current_line_[before_index]`.
  struct MidLineComment {
    int before_index = 0;
    bool is_trailing = false;
    int blank_lines_before = 0;
    int blank_lines_after = 0;
    int end_byte = 0;
    Lex::CommentIndex comment = Lex::CommentIndex::None;
    llvm::StringRef text;
  };

  const Parse::Tree* tree_;
  const Lex::TokenizedBuffer* tokens_;
  Style style_;

  // Collects each token's whitespace and generates the formatted text.
  WhitespaceManager whitespace_;
  std::string output_;

  // Inclusive 1-based source-line extents `(first_line, last_line)` of flushed
  // unwrapped lines and aligned trailing-comment groups, used by
  // `AffectedByteRanges`.
  llvm::SmallVector<std::pair<int, int>> coupled_line_extents_;

  llvm::SmallVector<TokenSpan> token_map_;

  // Per-token formatting annotations derived from `tree_`.
  TokenInfoStore token_infos_;

  // Tokens buffered for the current unwrapped line.
  llvm::SmallVector<Lex::TokenIndex> current_line_;

  // Comments encountered inside `current_line_`, emitted during `FlushLine()`.
  llvm::SmallVector<MidLineComment> mid_line_comments_;

  // Source byte offset past the last emitted token or comment, used to detect
  // blank lines and same-line trailing comments.
  std::optional<int> last_end_byte_;

  // True when the last rendered line ended with `{`, suppressing blank lines
  // at the start of a block.
  bool after_open_brace_ = false;

  // True once any token or comment has been recorded into `whitespace_`.
  bool started_ = false;

  // Current block indentation in spaces.
  int indent_ = 0;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
