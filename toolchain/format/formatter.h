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
// parse-tree annotations for spacing, line termination, and line wrapping.
class Formatter {
 public:
  explicit Formatter(const Parse::Tree* tree, const Style& style);

  // Formats into the internal buffer and returns true if the input had no lex
  // or parse errors. Must be called once before `TakeOutput`.
  auto Run() -> bool;

  // Returns the full formatted text, leaving the formatter empty.
  auto TakeOutput() -> std::string { return std::move(output_); }

 private:
  auto VerbatimGapBefore(Lex::TokenIndex token) const
      -> std::optional<llvm::StringRef>;

  auto FlushLine() -> void;

  auto PushToken(Lex::TokenIndex token) -> void;

  auto ShouldFlushBeforeComments(Lex::TokenIndex token) const -> bool;

  auto ClampBlankLines(llvm::StringRef gap) const -> int;

  auto IsAttachedTrailingComment(llvm::StringRef source, int prev_end,
                                 int start_byte,
                                 Lex::CommentIndex comment) const -> bool;

  auto ComputeBlankLines(int next_start_byte, bool is_block_end) -> int;

  auto LeadingNewlines(int next_start_byte, bool is_block_end) -> int {
    return (started_ ? 1 : 0) +
           ComputeBlankLines(next_start_byte, is_block_end);
  }

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

  WhitespaceManager whitespace_;
  std::string output_;

  TokenInfoStore token_infos_;
  llvm::SmallVector<Lex::TokenIndex> current_line_;
  llvm::SmallVector<MidLineComment> mid_line_comments_;

  std::optional<int> last_end_byte_;
  bool after_open_brace_ = false;
  bool started_ = false;
  int indent_ = 0;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
