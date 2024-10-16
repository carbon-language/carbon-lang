// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
#define CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_

#include <cstdint>

#include "common/ostream.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// Implements Format(); see format.h. It's intended to be constructed and
// `Run()` once, then destructed.
//
// TODO: This will probably need to work less linearly in the future, for
// example to handle smart wrapping of arguments. This is a simple
// implementation that only handles simple code. Before adding too much more
// complexity, it should be rewritten.
//
// TODO: Add retention of blank lines between original code.
//
// TODO: Add support for formatting line ranges (will need flags too).
class Formatter {
 public:
  explicit Formatter(const Lex::TokenizedBuffer* tokens, llvm::raw_ostream* out)
      : tokens_(tokens), out_(out) {}

  // See class comments.
  auto Run() -> bool;

 private:
  // Tracks the status of the current line of output.
  enum class LineState : uint8_t {
    // There is no output for the current line.
    Empty,
    // The current line has content (possibly just an indent), and does not need
    // a separator added.
    HasSeparator,
    // The current line has content, and will need a separator, typically a
    // single space or newline.
    NeedsSeparator,
  };

  // Ensure output is on an empty line, setting line_state_ to Empty. May output
  // a newline, dependent on line state. Does not indent, allowing blank lines.
  auto RequireEmptyLine() -> void;

  // Ensures there is a separator before adding new content. May do
  // `PrepareForPackedContent` or output a separator space, dependent on line
  // state. Always results in line_state_ being HasSeparator; the caller is
  // responsible for adjusting state if needed.
  auto PrepareForSpacedContent() -> void;

  // Requires that the current line is indented, but not necessarily a separator
  // space. May output spaces for `indent_`, dependent on line state. Only
  // guarantees the line_state_ is not Empty; the caller is responsible for
  // adjusting state if needed.
  auto PrepareForPackedContent() -> void;

  // Returns the next token index.
  static auto NextToken(Lex::TokenIndex token) -> Lex::TokenIndex {
    return *(Lex::TokenIterator(token) + 1);
  }

  // The tokens being formatted.
  const Lex::TokenizedBuffer* tokens_;

  // The output stream for formatted content.
  llvm::raw_ostream* out_;

  // The state of the line currently written to output.
  LineState line_state_ = LineState::Empty;

  // The current code indent level, to be added to new lines.
  int indent_ = 0;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
