// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/formatter.h"

#include <cstdlib>

namespace Carbon::Format {

auto Formatter::Run() -> bool {
  if (tokens_->has_errors()) {
    // TODO: Error recovery.
    return false;
  }

  auto comments = tokens_->comments();
  auto comment_it = comments.begin();

  // Scan for blank lines in the original source to preserve formatting.
  // A blank line is a line that has no tokens or comments.
  int prev_line = -1;
  for (auto token : tokens_->tokens()) {
    auto line = tokens_->GetLine(token);
    if (line != prev_line + 1 && prev_line >= 0) {
      // There was a gap in lines, mark all intermediate lines as blank.
      for (int i = prev_line + 1; i < line; ++i) {
        blank_lines_.insert(i);
      }
    }
    prev_line = line;
  }

  // If there are no tokens or comments, format as empty.
  if (tokens_->size() == 0 && comment_it == comments.end()) {
    *out_ << "\n";
    output_line_ = 1;
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
      ++output_line_;
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
    ++output_line_;
    line_state_ = LineState::Empty;
  }
  // Check if there was a blank line in the original source at this output line.
  // This preserves blank lines between code blocks.
  if (blank_lines_.count(output_line_) > 0) {
    *out_ << "\n";
    ++output_line_;
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
