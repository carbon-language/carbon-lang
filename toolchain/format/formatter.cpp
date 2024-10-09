// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/formatter.h"

namespace Carbon::Format {

auto Formatter::Run() -> bool {
  if (tokens_->has_errors()) {
    // TODO: Error recovery.
    return false;
  }

  auto comments = tokens_->comments();
  auto comment_it = comments.begin();

  // If there are no tokens or comments, format as empty.
  if (tokens_->size() == 0 && comment_it == comments.end()) {
    *out_ << "\n";
    return true;
  }

  for (auto token : tokens_->tokens()) {
    auto token_kind = tokens_->GetKind(token);

    while (comment_it != comments.end() &&
           tokens_->IsAfterComment(token, *comment_it)) {
      AddNewline();
      AddWhitespace();
      // TODO: We do need to adjust the indent of multi-line comments.
      *out_ << tokens_->GetCommentText(*comment_it);
      // Comment text includes a terminating newline, so just update the state.
      line_state_ = LineState::Empty;
      ++comment_it;
    }

    switch (token_kind) {
      case Lex::TokenKind::FileStart:
        break;

      case Lex::TokenKind::FileEnd:
        AddNewline();
        break;

      case Lex::TokenKind::OpenCurlyBrace:
        AddWhitespace();
        *out_ << "{";
        // Check for `{}`.
        if (NextToken(token) != tokens_->GetMatchedClosingToken(token)) {
          AddNewline();
        }
        indent_ += 2;
        break;

      case Lex::TokenKind::CloseCurlyBrace:
        indent_ -= 2;
        AddIndent();
        *out_ << "}";
        AddNewline();
        break;

      case Lex::TokenKind::Semi:
        AddIndent();
        *out_ << ";";
        AddNewline();
        break;

      default:
        if (token_kind.IsOneOf(
                {Lex::TokenKind::CloseParen, Lex::TokenKind::Colon,
                 Lex::TokenKind::ColonExclaim, Lex::TokenKind::Comma})) {
          AddIndent();
        } else {
          AddWhitespace();
        }
        *out_ << tokens_->GetTokenText(token);
        if (!token_kind.is_opening_symbol()) {
          line_state_ = LineState::WantsSpace;
        }
        break;
    }
  }
  return true;
}

auto Formatter::AddIndent() -> void {
  if (line_state_ == LineState::Empty) {
    out_->indent(indent_);
    line_state_ = LineState::HasContent;
  }
}

auto Formatter::AddNewline() -> void {
  if (line_state_ != LineState::Empty) {
    *out_ << "\n";
    line_state_ = LineState::Empty;
  }
}

auto Formatter::AddWhitespace() -> void {
  if (line_state_ == LineState::WantsSpace) {
    *out_ << " ";
    line_state_ = LineState::HasContent;
  } else {
    AddIndent();
  }
}

}  // namespace Carbon::Format
