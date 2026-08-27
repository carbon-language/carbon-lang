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

  // If there are no tokens or comments, format as empty.
  if (tokens_->size() == 0 && next_comment_ == comments_end_) {
    *out_ << "\n";
    return true;
  }

  Lex::TokenKind prev_token_kind = Lex::TokenKind::FileStart;
  for (auto token : tokens_->tokens()) {
    auto token_kind = tokens_->GetKind(token);

    // Emit any comments that come before this token in the source. Trailing
    // comments are attached to the still-open current line; full-line comments
    // are emitted on their own line.
    while (next_comment_ != comments_end_ &&
           tokens_->IsAfterComment(token, *next_comment_)) {
      EmitComment();
    }

    int token_start_line = tokens_->GetLineNumber(token);

    switch (token_kind) {
      case Lex::TokenKind::FileStart:
        break;

      case Lex::TokenKind::FileEnd:
        RequireEmptyLine();
        break;

      case Lex::TokenKind::OpenCurlyBrace:
        PrepareForSpacedContent(token_start_line);
        *out_ << "{";
        // Check for `{}`.
        if (NextToken(token) != tokens_->GetMatchedClosingToken(token)) {
          RequireEmptyLine();
        }
        indent_ += 2;
        break;

      case Lex::TokenKind::CloseCurlyBrace:
        indent_ -= 2;
        PrepareForPackedContent(token_start_line);
        *out_ << "}";
        RequireEmptyLine();
        break;

      case Lex::TokenKind::Else:
        if (prev_token_kind == Lex::TokenKind::CloseCurlyBrace &&
            line_state_ == LineState::EndOfLine) {
          line_state_ = LineState::NeedsSeparator;
        }
        PrepareForSpacedContent(token_start_line);
        *out_ << "else";
        line_state_ = LineState::NeedsSeparator;
        break;

      case Lex::TokenKind::Period:
        PrepareForPackedContent(token_start_line);
        *out_ << ".";
        line_state_ = LineState::HasSeparator;
        break;

      case Lex::TokenKind::PlusPlus:
      case Lex::TokenKind::MinusMinus:
        PrepareForSpacedContent(token_start_line);
        *out_ << tokens_->GetTokenText(token);
        line_state_ = LineState::HasSeparator;
        break;

      case Lex::TokenKind::Semi:
        PrepareForPackedContent(token_start_line);
        *out_ << ";";
        RequireEmptyLine();
        break;

      default:
        if (token_kind.IsOneOf(
                {Lex::TokenKind::CloseParen, Lex::TokenKind::CloseSquareBracket,
                 Lex::TokenKind::Colon, Lex::TokenKind::Comma})) {
          PrepareForPackedContent(token_start_line);
        } else if (token_kind.IsOneOf({Lex::TokenKind::OpenParen,
                                       Lex::TokenKind::OpenSquareBracket}) &&
                   (prev_token_kind.IsOneOf(
                        {Lex::TokenKind::Identifier, Lex::TokenKind::Array,
                         Lex::TokenKind::CloseParen,
                         Lex::TokenKind::CloseSquareBracket}) ||
                    prev_token_kind.is_sized_type_literal())) {
          PrepareForPackedContent(token_start_line);
        } else {
          PrepareForSpacedContent(token_start_line);
        }
        *out_ << tokens_->GetTokenText(token);
        line_state_ = token_kind.is_opening_symbol()
                          ? LineState::HasSeparator
                          : LineState::NeedsSeparator;
        break;
    }
    prev_token_kind = token_kind;
    if (token_kind != Lex::TokenKind::FileStart) {
      prev_end_line_ = tokens_->GetEndLoc(token).first.index + 1;
    }
  }

  // Materialize any newline deferred by the final line.
  if (line_state_ == LineState::EndOfLine) {
    *out_ << "\n";
    line_state_ = LineState::Empty;
  }
  return true;
}

auto Formatter::EmitComment() -> void {
  auto comment = *next_comment_;
  ++next_comment_;

  if (tokens_->IsTrailingComment(comment) && line_state_ != LineState::Empty) {
    // Keep the trailing comment on the current line, separated by a space. The
    // line still has content because its newline was deferred (`EndOfLine`) or
    // not yet required.
    *out_ << " " << tokens_->GetCommentText(comment);
    prev_end_line_ = tokens_->GetLineNumber(comment);
  } else {
    // A full-line comment (or a trailing comment with nothing left to attach
    // to) is emitted on its own line.
    RequireEmptyLine();
    int comment_start_line = tokens_->GetLineNumber(comment);
    PrepareForStartOfLine(comment_start_line);
    llvm::StringRef remaining = tokens_->GetCommentText(comment);
    bool first_line = true;
    while (!remaining.empty()) {
      auto [line, rest] = remaining.split('\n');
      remaining = rest;
      if (!first_line) {
        out_->indent(indent_);
      }
      first_line = false;
      *out_ << line.ltrim() << "\n";
    }
    int comment_lines = tokens_->GetCommentText(comment).count('\n');
    prev_end_line_ =
        comment_start_line + (comment_lines > 0 ? comment_lines - 1 : 0);
  }
  // Comment text includes a terminating newline, so just update the state.
  line_state_ = LineState::Empty;
}

auto Formatter::PrepareForStartOfLine(int start_line) -> void {
  // Materialize a deferred newline before starting to fill a fresh line.
  if (line_state_ == LineState::EndOfLine) {
    if (prev_end_line_ > 0 && start_line - prev_end_line_ >= 2) {
      *out_ << "\n\n";
    } else {
      *out_ << "\n";
    }
    line_state_ = LineState::Empty;
  } else if (line_state_ == LineState::Empty) {
    if (prev_end_line_ > 0 && start_line - prev_end_line_ >= 2) {
      *out_ << "\n";
    }
  }
  if (line_state_ == LineState::Empty) {
    out_->indent(indent_);
    line_state_ = LineState::HasSeparator;
  }
}

auto Formatter::PrepareForPackedContent(int start_line) -> void {
  PrepareForStartOfLine(start_line);
}

auto Formatter::RequireEmptyLine() -> void {
  // Defer the newline so a trailing comment can still attach to this line; it
  // is materialized by the next content or at end of file.
  if (line_state_ != LineState::Empty) {
    line_state_ = LineState::EndOfLine;
  }
}

auto Formatter::PrepareForSpacedContent(int start_line) -> void {
  if (line_state_ == LineState::NeedsSeparator) {
    *out_ << " ";
    line_state_ = LineState::HasSeparator;
  } else {
    PrepareForPackedContent(start_line);
  }
}

}  // namespace Carbon::Format
