// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/formatter.h"

#include "common/check.h"
#include "toolchain/format/comment.h"
#include "toolchain/format/line_wrapper.h"

namespace Carbon::Format {

Formatter::Formatter(const Parse::Tree* tree, llvm::raw_ostream* out)
    : tree_(tree),
      tokens_(&tree->tokens()),
      out_(out),
      comment_it_(tokens_->comments().begin()),
      comments_end_(tokens_->comments().end()),
      token_infos_(
          TokenInfoStore::MakeWithExplicitSize(tokens_->size(), TokenInfo())) {
  // Derive per-token formatting data from the tokens and the parse tree. Each
  // token's width is cached first. The role of each owned token then comes
  // from its node; multiple nodes can map to one token (virtual tokens, error
  // trees), so only a distinguishing role is recorded and the rest keep the
  // default `Unknown`. Binary-operator nodes additionally contribute the
  // operator-precedence data that drives operand alignment and break
  // penalties: each such node's operator token gets a break-after penalty, and
  // each operand-aligning operator opens an alignment scope at the first token
  // of its operand span and closes it at the last.
  //
  // Operand spans are the operator nodes' subtree token ranges, computed
  // within the same postorder pass: a stack of completed subtree ranges merges
  // children into parents in linear time (each node pops exactly its children,
  // identified by its kind's child count or bracketing node kind, the same
  // walk `Parse::TreeAndSubtrees` uses to compute subtree sizes), where
  // calling `GetSubtreeTokenRange` per operator node would walk each operand
  // subtree again (quadratic for a long operator chain).
  for (auto token : tokens_->tokens()) {
    // A multi-line token (such as a multi-line string literal) occupies its
    // first physical line where it sits; the rest take their own lines. So its
    // width on the current line is the first line's, not the whole text.
    token_infos_.Get(token).column_width =
        static_cast<int>(tokens_->GetTokenText(token).split('\n').first.size());
  }
  // A completed subtree's root node kind (used to match its parent's
  // bracketing node kind) and inclusive token range; the range is `None` for
  // a subtree with no valued tokens.
  struct SubtreeRange {
    Parse::NodeKind kind;
    Lex::TokenIndex min = Lex::TokenIndex::None;
    Lex::TokenIndex max = Lex::TokenIndex::None;
  };
  llvm::SmallVector<SubtreeRange> range_stack;
  for (auto node_id : tree_->postorder()) {
    auto kind = tree_->node_kind(node_id);
    Lex::TokenIndex token = tree_->node_token(node_id);
    TokenRole role = RoleForNodeKind(kind);
    if (role != TokenRole::Unknown && token.has_value()) {
      token_infos_.Get(token).role = role;
    }

    // Fold the node's own token and its children's completed ranges into its
    // range, popping exactly the children off the stack: a fixed number for a
    // node kind with a child count, or entries back through the bracketing
    // node kind otherwise.
    SubtreeRange range = {.kind = kind};
    if (token.has_value()) {
      range.min = range.max = token;
    }
    auto fold_child = [&]() -> Parse::NodeKind {
      CARBON_CHECK(!range_stack.empty(), "NodeId {0} ({1}) is missing children",
                   node_id, kind);
      SubtreeRange child = range_stack.pop_back_val();
      if (child.min.has_value() &&
          (!range.min.has_value() || child.min < range.min)) {
        range.min = child.min;
      }
      if (child.max.has_value() &&
          (!range.max.has_value() || child.max > range.max)) {
        range.max = child.max;
      }
      return child.kind;
    };
    if (kind.has_child_count()) {
      for (int i = 0; i < kind.child_count(); ++i) {
        fold_child();
      }
    } else {
      while (fold_child() != kind.bracket()) {
      }
    }

    auto op_info = OperatorInfoForNodeKind(kind);
    if (op_info.break_penalty >= 0 && token.has_value()) {
      token_infos_.Get(token).break_penalty_after = op_info.break_penalty;
      if (op_info.aligns_operands) {
        ++token_infos_.Get(range.min).open_scopes;
        ++token_infos_.Get(range.max).close_scopes;
      }
    }
    range_stack.push_back(range);
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

  Lex::TokenIndex last = current_line_.back();
  last_end_byte_ =
      tokens_->GetByteOffset(last) + tokens_->GetTokenText(last).size();
  after_open_brace_ = tokens_->GetKind(last) == Lex::TokenKind::OpenCurlyBrace;
  // Keep any trailing comment on this line, then end it.
  AttachTrailingComments();
  *out_ << "\n";
  current_line_.clear();
}

auto Formatter::AttachTrailingComments() -> void {
  llvm::StringRef source = tokens_->source().text();
  while (comment_it_ != comments_end_ &&
         tokens_->IsTrailingComment(*comment_it_)) {
    llvm::StringRef text = tokens_->GetCommentText(*comment_it_);
    int start_byte = text.data() - source.data();
    // A trailing comment attaches only if it directly follows the code just
    // rendered: nothing but horizontal whitespace between the last token and
    // the comment. Checking for a line break alone is not enough: with
    // several statements on one source line, the comment must attach to the
    // last of them, not to the first to be flushed.
    if (!source.slice(*last_end_byte_, start_byte).trim(" \t").empty()) {
      break;
    }
    *out_ << " " << text.rtrim();
    last_end_byte_ = start_byte + text.rtrim().size();
    after_open_brace_ = false;
    ++comment_it_;
  }
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
  for (auto token : tokens_->tokens()) {
    // Emit any comments that sort before this token. A full-line comment block
    // is re-indented to the current code indent and wrapped to the column
    // limit; a trailing comment stays on the line of the code it follows.
    while (comment_it_ != comments_end_ &&
           tokens_->IsAfterComment(token, *comment_it_)) {
      if (tokens_->IsTrailingComment(*comment_it_) && !current_line_.empty()) {
        // The comment trails code still buffered in the current line (a comment
        // mid-way through a wrapped statement); flush it so the comment
        // attaches to that line. A trailing comment at the end of a statement
        // was already attached when its terminator flushed the line, so it
        // never reaches here. If the comment cannot attach (no open line), it
        // falls through to own-line emission below.
        FlushLine();
        continue;
      }
      FlushLine();
      llvm::StringRef text = tokens_->GetCommentText(*comment_it_);
      int start_byte = text.data() - tokens_->source().text().data();
      MaybeBlankLine(start_byte, /*is_block_end=*/false);
      *out_ << CommentText(text, indent_, ColumnLimit) << "\n";
      // Comment text includes its trailing newline (though a comment ending
      // the file may lack it); exclude it from the byte baseline so a
      // following blank line is counted consistently with tokens (whose text
      // has no trailing newline).
      last_end_byte_ = start_byte + text.rtrim().size();
      after_open_brace_ = false;
      ++comment_it_;
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
        bool has_inner_comment = comment_it_ != comments_end_ &&
                                 tokens_->GetCommentText(*comment_it_).data() -
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
