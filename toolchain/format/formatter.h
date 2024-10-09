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
class Formatter {
 public:
  explicit Formatter(const Lex::TokenizedBuffer* tokens, llvm::raw_ostream* out)
      : tokens_(tokens), out_(out) {}

  // See class comments.
  //
  // TODO: This will probably need to work less linearly in the future, for
  // example to handle smart wrapping of arguments. This is a simple
  // implementation that only handles simple code. Before adding too much more
  // complexity, it should be rewritten.
  //
  // TODO: Add retention of blank lines between original code.
  //
  // TODO: Add support for formatting line ranges (will need flags too).
  auto Run() -> bool;

 private:
  // Tracks the status of the current line of output.
  enum class LineState : uint8_t {
    // There is no output for the current line.
    Empty,
    // The current line has been indented, and may have text.
    HasContent,
    // If more output is added to the current line, add a space first.
    WantsSpace,
  };

  // May indent output, dependent on line state.
  auto AddIndent() -> void;

  // May output a newline, dependent on line state.
  auto AddNewline() -> void;

  // May indent output or just add a space, dependent on line state.
  auto AddWhitespace() -> void;

  // Returns the next token index.
  static auto NextToken(Lex::TokenIndex token) -> Lex::TokenIndex {
    return *(Lex::TokenIterator(token) + 1);
  }

  const Lex::TokenizedBuffer* tokens_;
  llvm::raw_ostream* out_;
  LineState line_state_ = LineState::Empty;
  int indent_ = 0;
};

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_FORMATTER_H_
