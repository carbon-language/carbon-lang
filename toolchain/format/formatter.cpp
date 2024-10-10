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
      RequireEmptyLine();
      PrepareForSpacedContent();
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
        RequireEmptyLine();
        break;

      case Lex::TokenKind::OpenCurlyBrace:
        PrepareForSpacedContent();
        *out_ << "{";
        // Check for `{}`.
        if (NextToken(token) != tokens_->GetMatchedClosingToken(token)) {
          RequireEmptyLine();
        }
        indent_ += 2;
        break;

      case Lex::TokenKind::CloseCurlyBrace:
        indent_ -= 2;
        PrepareForPackedContent();
        *out_ << "}";
        RequireEmptyLine();
        break;

      case Lex::TokenKind::Semi:
        PrepareForPackedContent();
        *out_ << ";";
        RequireEmptyLine();
        break;

      default:
        if (token_kind.IsOneOf(
                {Lex::TokenKind::CloseParen, Lex::TokenKind::Colon,
                 Lex::TokenKind::ColonExclaim, Lex::TokenKind::Comma})) {
          PrepareForPackedContent();
        } else {
          PrepareForSpacedContent();
        }
        *out_ << tokens_->GetTokenText(token);
        line_state_ = token_kind.is_opening_symbol()
                          ? LineState::HasSeparator
                          : LineState::NeedsSeparator;
        break;
    }
  }
  return true;
}

auto Formatter::PrepareForPackedContent() -> void {
  if (line_state_ == LineState::Empty) {
    out_->indent(indent_);
    line_state_ = LineState::HasSeparator;
  }
}

auto Formatter::RequireEmptyLine() -> void {
  if (line_state_ != LineState::Empty) {
    *out_ << "\n";
    line_state_ = LineState::Empty;
  }
}

auto Formatter::PrepareForSpacedContent() -> void {
  if (line_state_ == LineState::NeedsSeparator) {
    *out_ << " ";
    line_state_ = LineState::HasSeparator;
  } else {
    PrepareForPackedContent();
  }
}

}  // namespace Carbon::Format
