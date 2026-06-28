// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_

#include <optional>

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/token_info.h"
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
// TODO: Drive indentation and line structure from the parse tree rather than
// from brace nesting.
//
// TODO: Add support for formatting line ranges (will need flags too).
class Formatter {
 public:
  explicit Formatter(const Parse::Tree* tree, llvm::raw_ostream* out);

  // See class comments.
  auto Run() -> bool;

 private:
  // Renders the buffered line (if any) at the current indent, with inter-token
  // spacing and any preceding blank line, then ends it and clears the buffer.
  auto FlushLine() -> void;

  // Emits a single blank line if the original source had one or more blank
  // lines before the content starting at `next_start_byte`, unless that would
  // fall at the start or end of a block. At most one blank line is kept.
  auto MaybeBlankLine(int next_start_byte, bool is_block_end) -> void;

  // Returns the next token index.
  static auto NextToken(Lex::TokenIndex token) -> Lex::TokenIndex {
    return *(Lex::TokenIterator(token) + 1);
  }

  // The parse tree being formatted.
  const Parse::Tree* tree_;

  // The tokens being formatted, referenced by the parse tree.
  const Lex::TokenizedBuffer* tokens_;

  // The output stream for formatted content.
  llvm::raw_ostream* out_;

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

  // The current code indent level, in spaces, added to new lines.
  int indent_ = 0;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
