// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/format.h"

#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// Returns the next token index.
static auto NextToken(Lex::TokenIndex token) -> Lex::TokenIndex {
  return *(Lex::TokenIterator(token) + 1);
}

// Tracks the status of the current line of output.
enum class LineState : uint8_t {
  // There is no output for the current line.
  Empty,
  // The current line has been indented, and may have more text.
  Indented,
  // If more output is added to the current line, add a space first.
  WantsSpace,
};

// Adds a newline when needed.
static auto AddNewline(llvm::raw_ostream& out, LineState& line_state) -> void {
  if (line_state != LineState::Empty) {
    out << "\n";
    line_state = LineState::Empty;
  }
}

// Adds an indent if needed.
static auto AddIndent(llvm::raw_ostream& out, LineState& line_state, int indent)
    -> void {
  if (line_state == LineState::Empty) {
    out.indent(indent);
    line_state = LineState::Indented;
  }
}

static auto AddWhitespace(llvm::raw_ostream& out, LineState& line_state,
                          int indent) -> void {
  if (line_state == LineState::WantsSpace) {
    out << " ";
    line_state = LineState::Indented;
  } else {
    AddIndent(out, line_state, indent);
  }
}

// TODO: This will probably need to work less linearly in the future, for
// example to handle smart wrapping of arguments. This is a simple
// implementation that only handles simple code. Before adding too much more
// complexity, it should be rewritten.
//
// TODO: Add retention of blank lines between original code.
//
// TODO: Add support for formatting line ranges (will need flags too).
auto Format(const Lex::TokenizedBuffer& tokens, llvm::raw_ostream& out)
    -> bool {
  if (tokens.has_errors()) {
    // TODO: Error recovery.
    return false;
  }

  auto comments = tokens.comments();
  auto comment_it = comments.begin();

  // If there are no tokens or comments, format as empty.
  if (tokens.size() == 0 && comment_it == comments.end()) {
    out << "\n";
    return true;
  }

  int indent = 0;
  auto line_state = LineState::Empty;

  for (auto token : tokens.tokens()) {
    auto token_kind = tokens.GetKind(token);

    while (comment_it != comments.end() &&
           tokens.IsAfterComment(token, *comment_it)) {
      AddNewline(out, line_state);
      AddWhitespace(out, line_state, indent);
      // TODO: We do need to adjust the indent of multi-line comments.
      out << tokens.GetCommentText(*comment_it);
      // Comment text includes a terminating newline, so just update the state.
      line_state = LineState::Empty;
      ++comment_it;
    }

    switch (token_kind) {
      case Lex::TokenKind::FileStart:
        break;

      case Lex::TokenKind::FileEnd:
        AddNewline(out, line_state);
        break;

      case Lex::TokenKind::OpenCurlyBrace:
        AddWhitespace(out, line_state, indent);
        out << "{";
        // Check for `{}`.
        if (NextToken(token) != tokens.GetMatchedClosingToken(token)) {
          AddNewline(out, line_state);
        }
        indent += 2;
        break;

      case Lex::TokenKind::CloseCurlyBrace:
        indent -= 2;
        AddIndent(out, line_state, indent);
        out << "}";
        AddNewline(out, line_state);
        break;

      case Lex::TokenKind::Semi:
        AddIndent(out, line_state, indent);
        out << ";";
        AddNewline(out, line_state);
        break;

      default:
        if (token_kind.IsOneOf(
                {Lex::TokenKind::CloseParen, Lex::TokenKind::Colon,
                 Lex::TokenKind::ColonExclaim, Lex::TokenKind::Comma})) {
          AddIndent(out, line_state, indent);
        } else {
          AddWhitespace(out, line_state, indent);
        }
        out << tokens.GetTokenText(token);
        if (!token_kind.is_opening_symbol()) {
          line_state = LineState::WantsSpace;
        }
        break;
    }
  }
  return true;
}

}  // namespace Carbon::Format
