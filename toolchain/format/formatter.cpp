// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/formatter.h"

namespace Carbon::Format {

Formatter::Formatter(const Parse::Tree* tree, llvm::raw_ostream* out)
    : tree_(tree),
      tokens_(&tree->tokens()),
      out_(out),
      token_infos_(
          TokenInfoStore::MakeWithExplicitSize(tokens_->size(), TokenInfo())) {
  // Derive each token's role from the parse node it is the root of. Multiple
  // nodes can map to one token (virtual tokens, error trees), so only a
  // distinguishing role is recorded and the rest keep the default `Unknown`.
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

auto Formatter::EmitToken(Lex::TokenIndex token) -> void {
  Lex::TokenKind kind = tokens_->GetKind(token);
  int start_byte = tokens_->GetByteOffset(token);
  if (at_line_start_) {
    MaybeBlankLine(start_byte, kind == Lex::TokenKind::CloseCurlyBrace);
    out_->indent(indent_);
    at_line_start_ = false;
  } else if (previous_) {
    out_->indent(SpacesBefore(*tokens_, token_infos_, *previous_, token));
  }
  llvm::StringRef text = tokens_->GetTokenText(token);
  *out_ << text;
  previous_ = token;
  last_end_byte_ = start_byte + text.size();
  after_open_brace_ = kind == Lex::TokenKind::OpenCurlyBrace;
}

auto Formatter::Newline() -> void {
  *out_ << "\n";
  at_line_start_ = true;
  previous_ = std::nullopt;
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
    // Emit any comments that sort before this token.
    while (comment_it != comments.end() &&
           tokens_->IsAfterComment(token, *comment_it)) {
      if (!at_line_start_) {
        Newline();
      }
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
      at_line_start_ = true;
      previous_ = std::nullopt;
      last_end_byte_ = start_byte + text.rtrim().size();
      after_open_brace_ = false;
      ++comment_it;
    }

    switch (tokens_->GetKind(token)) {
      case Lex::TokenKind::FileStart:
        break;

      case Lex::TokenKind::FileEnd:
        // Ensure the file ends with a newline if it has trailing content. An
        // empty file stays empty.
        if (!at_line_start_) {
          Newline();
        }
        break;

      case Lex::TokenKind::OpenCurlyBrace:
        EmitToken(token);
        // Expand a non-empty block onto its own lines; keep `{}` compact.
        if (NextToken(token) != tokens_->GetMatchedClosingToken(token)) {
          Newline();
        }
        indent_ += 2;
        break;

      case Lex::TokenKind::CloseCurlyBrace: {
        indent_ -= 2;
        // Put the close brace of a non-empty block on its own line. For valid
        // code the previous token already ended the line; this handles
        // best-effort cases (such as a missing `;`) where it didn't.
        auto open = tokens_->GetMatchedOpeningToken(token);
        if (!at_line_start_ && NextToken(open) != token) {
          Newline();
        }
        EmitToken(token);
        // A separator, an `=`, or `else` continues the close-brace line
        // (`};`, `} else {`) rather than starting its own; anything else
        // begins a new line.
        if (!tokens_->GetKind(NextToken(token))
                 .IsOneOf({Lex::TokenKind::Semi, Lex::TokenKind::Comma,
                           Lex::TokenKind::CloseParen,
                           Lex::TokenKind::CloseSquareBracket,
                           Lex::TokenKind::Equal, Lex::TokenKind::Else})) {
          Newline();
        }
        break;
      }

      case Lex::TokenKind::Semi:
        EmitToken(token);
        Newline();
        break;

      default:
        EmitToken(token);
        break;
    }
  }
  return !tokens_->has_errors() && !tree_->has_errors();
}

}  // namespace Carbon::Format
