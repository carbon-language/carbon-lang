// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/formatter.h"

#include "toolchain/format/line_wrapper.h"

namespace Carbon::Format {

Formatter::Formatter(const Parse::Tree* tree, llvm::raw_ostream* out)
    : tree_(tree),
      tokens_(&tree->tokens()),
      out_(out),
      token_infos_(
          TokenInfoStore::MakeWithExplicitSize(tokens_->size(), TokenInfo())) {
  // Cache each token's width, then derive its role from the parse node it is
  // the root of. Multiple nodes can map to one token (virtual tokens, error
  // trees), so only a distinguishing role is recorded and the rest keep the
  // default `Unknown`.
  for (auto token : tokens_->tokens()) {
    // A multi-line token (such as a multi-line string literal) occupies its
    // first physical line where it sits; the rest take their own lines. So its
    // width on the current line is the first line's, not the whole text.
    token_infos_.Get(token).column_width =
        static_cast<int>(tokens_->GetTokenText(token).split('\n').first.size());
  }
  for (auto node_id : tree_->postorder()) {
    TokenRole role = RoleForNodeKind(tree_->node_kind(node_id));
    Lex::TokenIndex token = tree_->node_token(node_id);
    if (role != TokenRole::Unknown && token.has_value()) {
      token_infos_.Get(token).role = role;
    }
  }
}

auto Formatter::MaybeBlankLine(int next_start_byte, bool is_block_end) -> void {
  // The gap can be empty or inverted after a lexer-inserted recovery token,
  // whose synthesized byte offset can overlap the next real token; there is no
  // blank line to keep in that case.
  if (!last_end_byte_ || after_open_brace_ || is_block_end ||
      next_start_byte <= *last_end_byte_) {
    return;
  }
  llvm::StringRef gap = tokens_->source().text().substr(
      *last_end_byte_, next_start_byte - *last_end_byte_);
  // Two or more newlines between the previous content and this means there was
  // at least one blank line; keep a single one.
  if (gap.count('\n') >= 2) {
    *out_ << "\n";
  }
}

auto Formatter::FlushLine() -> void {
  if (current_line_.empty()) {
    return;
  }

  Lex::TokenIndex first = current_line_.front();
  MaybeBlankLine(tokens_->GetByteOffset(first),
                 tokens_->GetKind(first) == Lex::TokenKind::CloseCurlyBrace);

  // Decide where line breaks go. A line that already fits needs none: this is
  // both the common case and a fast path that keeps short output byte-for-byte
  // stable. A longer line goes to the wrapping solver.
  llvm::SmallVector<int> newline_indents;
  if (indent_ + RenderedWidth(*tokens_, token_infos_, current_line_) <=
      ColumnLimit) {
    newline_indents.assign(current_line_.size(), -1);
  } else {
    newline_indents =
        SolveLineBreaks(*tokens_, token_infos_, current_line_, indent_);
  }

  out_->indent(indent_);

  std::optional<Lex::TokenIndex> previous;
  for (int i = 0; i < static_cast<int>(current_line_.size()); ++i) {
    Lex::TokenIndex token = current_line_[i];
    if (previous) {
      if (newline_indents[i] >= 0) {
        // A line break before this token, then its continuation indent.
        *out_ << "\n";
        out_->indent(newline_indents[i]);
      } else {
        out_->indent(SpacesBefore(*tokens_, token_infos_, *previous, token));
      }
    }
    *out_ << tokens_->GetTokenText(token);
    previous = token;
  }
  *out_ << "\n";

  Lex::TokenIndex last = current_line_.back();
  last_end_byte_ =
      tokens_->GetByteOffset(last) + tokens_->GetTokenText(last).size();
  after_open_brace_ = tokens_->GetKind(last) == Lex::TokenKind::OpenCurlyBrace;
  current_line_.clear();
}

auto Formatter::Run() -> bool {
  // Format best-effort, even when the input has lex or parse errors: the parse
  // tree is always structurally valid, so output can always be produced. The
  // return value reports whether the input was error-free (so the driver can
  // reflect that in its exit code), but best-effort output is emitted either
  // way rather than giving up.
  //
  // TODO: For badly malformed regions, consider emitting the original source
  // verbatim (rather than reformatting) to better preserve author intent. This
  // requires identifying the minimal error subtrees, not just `has_errors()`.
  //
  // TODO: `//@...` tooling directive lines (`//@include-in-dumps` and
  // `//@dump-sem-ir-begin`/`-end`) are consumed by the lexer without being
  // recorded as comments, so output reconstructed from tokens and comments
  // silently drops them. Surface them through the lexer's comment records, or
  // detect and preserve them here.
  auto comments = tokens_->comments();
  auto comment_it = comments.begin();

  for (auto token : tokens_->tokens()) {
    // Emit any comments that sort before this token, each on its own line.
    while (comment_it != comments.end() &&
           tokens_->IsAfterComment(token, *comment_it)) {
      FlushLine();
      llvm::StringRef text = tokens_->GetCommentText(*comment_it);
      int start_byte = text.data() - tokens_->source().text().data();
      MaybeBlankLine(start_byte, /*is_block_end=*/false);
      out_->indent(indent_);
      // TODO: Re-indent multi-line comment bodies.
      *out_ << text;
      if (!text.ends_with('\n')) {
        // Only a comment ending the file without a final newline lacks one.
        *out_ << "\n";
      }
      // Comment text includes its trailing newline (added above if the source
      // lacked it); exclude it from the byte baseline so a following blank
      // line is counted consistently with tokens (whose text has no trailing
      // newline).
      last_end_byte_ = start_byte + text.rtrim().size();
      after_open_brace_ = false;
      ++comment_it;
    }

    switch (tokens_->GetKind(token)) {
      case Lex::TokenKind::FileStart:
        break;

      case Lex::TokenKind::FileEnd:
        // Render any trailing content. An empty file stays empty.
        FlushLine();
        break;

      case Lex::TokenKind::OpenCurlyBrace: {
        current_line_.push_back(token);
        // Expand the block onto its own lines unless it is empty, keeping `{}`
        // compact. A comment between the braces (which is not a token) counts
        // as content, so the block still expands.
        auto close = tokens_->GetMatchedClosingToken(token);
        bool has_inner_token = NextToken(token) != close;
        bool has_inner_comment = comment_it != comments.end() &&
                                 tokens_->GetCommentText(*comment_it).data() -
                                         tokens_->source().text().data() <
                                     tokens_->GetByteOffset(close);
        if (has_inner_token || has_inner_comment) {
          FlushLine();
        }
        indent_ += 2;
        break;
      }

      case Lex::TokenKind::CloseCurlyBrace:
        // If the line still holds content that isn't the matching open brace,
        // render it (at the inner indent, before dedenting) so the close brace
        // starts its own line. For valid code the line is already empty here;
        // this handles best-effort cases such as a missing `;`, while keeping
        // an empty `{}` compact.
        if (!current_line_.empty() &&
            current_line_.back() != tokens_->GetMatchedOpeningToken(token)) {
          FlushLine();
        }
        indent_ -= 2;
        current_line_.push_back(token);
        // A separator, an `=`, or `else` continues the close-brace line
        // (`};`, `} else {`) rather than starting its own, so only flush when
        // the next token starts a new line.
        if (!tokens_->GetKind(NextToken(token))
                 .IsOneOf({Lex::TokenKind::Semi, Lex::TokenKind::Comma,
                           Lex::TokenKind::CloseParen,
                           Lex::TokenKind::CloseSquareBracket,
                           Lex::TokenKind::Equal, Lex::TokenKind::Else})) {
          FlushLine();
        }
        break;

      case Lex::TokenKind::Semi:
        current_line_.push_back(token);
        FlushLine();
        break;

      default:
        current_line_.push_back(token);
        break;
    }
  }
  return !tokens_->has_errors() && !tree_->has_errors();
}

}  // namespace Carbon::Format
