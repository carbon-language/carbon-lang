// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/formatter.h"

#include <algorithm>

#include "common/check.h"
#include "common/map.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/format/comment.h"
#include "toolchain/format/cpp_snippet.h"
#include "toolchain/format/line_wrapper.h"

namespace Carbon::Format {

Formatter::Formatter(const Parse::Tree* tree, const Style& style)
    : tree_(tree),
      tokens_(&tree->tokens()),
      style_(style),
      whitespace_(tokens_, style),
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
  // The string literal body of an `inline Cpp` (or `import Cpp inline`)
  // declaration holds C++ code; see the postorder pass below.
  for (auto token : tokens_->tokens()) {
    // A multi-line token (such as a multi-line string literal) occupies its
    // first physical line where it sits; the rest take their own lines. So its
    // width on the current line is the first line's, not the whole text.
    token_infos_.Get(token).column_width =
        static_cast<int>(tokens_->GetTokenText(token).split('\n').first.size());
  }
  // The latest (outermost) member-access token seen in each chain, keyed by the
  // chain's receiver-root token index. The chain is left-nested, so postorder
  // visits its member accesses inner-to-outer; the last one seen is the final
  // link, which gets the cheaper break penalty.
  Map<int, int> chain_last_member;
  // A completed subtree's root node kind (used to match its parent's
  // bracketing node kind) and inclusive token range; the range is `None` for
  // a subtree with no valued tokens. `is_error_root` records that the
  // subtree's own node is erroneous and no enclosing subtree has claimed it
  // yet: if an erroneous ancestor folds it in, the ancestor's wider range
  // subsumes it; otherwise it is a maximal error subtree, the minimal span to
  // emit verbatim (see `MarkVerbatim` below).
  struct SubtreeRange {
    Parse::NodeKind kind;
    Lex::TokenIndex min = Lex::TokenIndex::None;
    Lex::TokenIndex max = Lex::TokenIndex::None;
    bool is_error_root = false;
  };
  llvm::SmallVector<SubtreeRange> range_stack;
  // Marks every token of a maximal error subtree, so the region is emitted
  // with its original source text; see `VerbatimGapBefore`. Maximal error
  // subtrees are disjoint, so the total marking work is linear.
  auto mark_verbatim = [&](const SubtreeRange& range) {
    if (!range.min.has_value()) {
      return;
    }
    has_verbatim_tokens_ = true;
    for (int index = range.min.index; index <= range.max.index; ++index) {
      token_infos_.Get(Lex::TokenIndex(index)).is_verbatim = true;
    }
  };
  for (auto node_id : tree_->postorder()) {
    auto kind = tree_->node_kind(node_id);
    Lex::TokenIndex token = tree_->node_token(node_id);
    TokenRole role = RoleForNodeKind(kind);
    if (role != TokenRole::Unknown && token.has_value()) {
      token_infos_.Get(token).role = role;
    }

    // The string literal body of an `inline Cpp` (or `import Cpp inline`)
    // declaration holds C++ code, reformatted regardless of its indicator.
    if (kind == Parse::NodeKind::InlineImportBody && token.has_value()) {
      token_infos_.Get(token).is_cpp_string = true;
    }

    // Fold the node's own token and its children's completed ranges into its
    // range, popping exactly the children off the stack: a fixed number for a
    // node kind with a child count, or entries back through the bracketing
    // node kind otherwise.
    SubtreeRange range = {.kind = kind};
    if (token.has_value()) {
      range.min = range.max = token;
    }
    bool node_has_error = tree_->node_has_error(node_id);
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
      // A child error subtree is subsumed when this node is erroneous too
      // (this node's wider range covers it); otherwise the child is a maximal
      // error subtree and its tokens go verbatim now.
      if (child.is_error_root && !node_has_error) {
        mark_verbatim(child);
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
    range.is_error_root = node_has_error;

    // A member-access node marks its `.`/`->` token as a break-before point
    // and joins it to its chain, identified by the receiver root: the first
    // token of the member-access subtree (`range.min`, just folded), shared
    // across the whole left-nested chain.
    int member_penalty = MemberAccessBreakPenalty(kind);
    if (member_penalty >= 0 && token.has_value()) {
      TokenInfo& info = token_infos_.Get(token);
      info.break_penalty_before = member_penalty;
      int chain_id = range.min.index;
      info.member_chain_id = chain_id;
      chain_last_member.Update(chain_id, token.index);
    }

    auto op_info = OperatorInfoForNodeKind(kind, style_);
    if (op_info.break_penalty >= 0 && token.has_value()) {
      token_infos_.Get(token).break_penalty_after = op_info.break_penalty;
      if (op_info.aligns_operands) {
        ++token_infos_.Get(range.min).open_scopes;
        ++token_infos_.Get(range.max).close_scopes;
      }
    }
    range_stack.push_back(range);
  }
  // Top-level subtrees have no parent to fold them, so any that are error
  // roots are maximal and go verbatim here.
  for (const SubtreeRange& range : range_stack) {
    if (range.is_error_root) {
      mark_verbatim(range);
    }
  }

  // The last link in each chain breaks more cheaply (clang-format's 35 vs 150),
  // so a chain that must wrap prefers to break at its end.
  chain_last_member.ForEach([&](int /*chain_id*/, int last_member) {
    token_infos_.Get(Lex::TokenIndex(last_member)).break_penalty_before =
        MemberAccessLastLinkBreakPenalty;
  });
}

auto Formatter::ComputeBlankLines(int next_start_byte, bool is_block_end)
    -> int {
  // The gap can be empty or inverted after a lexer-inserted recovery token,
  // whose synthesized byte offset can overlap the next real token; there is no
  // blank line to keep in that case.
  if (!last_end_byte_ || after_open_brace_ || is_block_end ||
      next_start_byte <= *last_end_byte_) {
    return 0;
  }
  llvm::StringRef gap = tokens_->source().text().substr(
      *last_end_byte_, next_start_byte - *last_end_byte_);
  // The blank lines between the previous content and this are the newlines in
  // the gap beyond the one that ends the previous line; keep up to the style's
  // maximum. A gap with no newline at all (content sharing the previous
  // source line) has no blank lines either; without the clamp the -1 would
  // cancel the structural line break in `LeadingNewlines` and glue the
  // content onto the previous line.
  return std::clamp(static_cast<int>(gap.count('\n')) - 1, 0,
                    style_.max_empty_lines_to_keep);
}

auto Formatter::VerbatimGapBefore(Lex::TokenIndex token) const
    -> std::optional<llvm::StringRef> {
  // A token's leading gap is copied verbatim when the token and the token
  // before it both lie in a verbatim error region, so the region's interior
  // keeps its original layout while its first token is still placed by the
  // formatter. A lexer-inserted recovery token has no real source position
  // (its synthesized offset can even overlap a neighbor), so a gap it bounds
  // cannot be sliced out of the source and is formatted normally instead.
  // The whole-file flag keeps this near-free on error-free input, the common
  // case, where it is checked for every token and comment.
  if (!has_verbatim_tokens_ || !token_infos_.Get(token).is_verbatim ||
      token.index == 0) {
    return std::nullopt;
  }
  Lex::TokenIndex prev(token.index - 1);
  if (!token_infos_.Get(prev).is_verbatim || tokens_->IsRecoveryToken(token) ||
      tokens_->IsRecoveryToken(prev)) {
    return std::nullopt;
  }
  int32_t prev_end = tokens_->GetByteOffset(prev) +
                     static_cast<int32_t>(tokens_->GetTokenText(prev).size());
  int32_t start = tokens_->GetByteOffset(token);
  if (start < prev_end) {
    return std::nullopt;
  }
  return tokens_->source().text().slice(prev_end, start);
}

auto Formatter::FlushLine() -> void {
  if (current_line_.empty()) {
    return;
  }

  Lex::TokenIndex first = current_line_.front();
  int leading_newlines = LeadingNewlines(
      tokens_->GetByteOffset(first),
      tokens_->GetKind(first) == Lex::TokenKind::CloseCurlyBrace);

  // Decide where line breaks go. A line that already fits needs none: this is
  // both the common case and a fast path that keeps short output byte-for-byte
  // stable. A longer line goes to the wrapping solver, except a line holding a
  // verbatim error region, whose original layout must not be re-wrapped.
  llvm::SmallVector<int> newline_indents;
  bool has_verbatim = has_verbatim_tokens_ &&
                      llvm::any_of(current_line_, [&](Lex::TokenIndex token) {
                        return token_infos_.Get(token).is_verbatim;
                      });
  if (has_verbatim ||
      indent_ + RenderedWidth(*tokens_, token_infos_, current_line_) <=
          style_.column_limit) {
    newline_indents.assign(current_line_.size(), -1);
  } else {
    newline_indents =
        SolveLineBreaks(*tokens_, token_infos_, current_line_, indent_, style_);
  }

  // Record each token's leading whitespace. The line's first token carries the
  // blank-line allowance and the break ending the previous line; a wrapped
  // token carries its own break and continuation indent; everything else its
  // inter-token spacing. The bracket-nesting depth is tracked for the alignment
  // pass, which uses it to tell a wrapped continuation line from a new
  // statement.
  std::optional<Lex::TokenIndex> previous;
  int nesting_level = 0;
  for (int i = 0; i < static_cast<int>(current_line_.size()); ++i) {
    Lex::TokenIndex token = current_line_[i];
    int newlines;
    int spaces;
    if (!previous) {
      newlines = leading_newlines;
      spaces = indent_;
    } else if (newline_indents[i] >= 0) {
      newlines = 1;
      spaces = newline_indents[i];
    } else {
      newlines = 0;
      spaces = SpacesBefore(*tokens_, token_infos_, *previous, token);
    }
    Lex::TokenKind kind = tokens_->GetKind(token);
    if (kind.IsOneOf({Lex::TokenKind::CloseParen,
                      Lex::TokenKind::CloseSquareBracket,
                      Lex::TokenKind::CloseCurlyBrace}) &&
        nesting_level > 0) {
      --nesting_level;
    }
    // A multi-line string literal that holds C++ (one with a `'''cpp` file
    // type indicator, or the body of an `inline Cpp` declaration) has its
    // body reformatted by clang-format and re-encoded in place; see
    // `cpp_snippet.h`. The body and closing delimiter indent to the statement.
    llvm::StringRef rewritten;
    std::optional<std::string> cpp_snippet;
    if (style_.format_cpp_snippets && !token_infos_.Get(token).is_verbatim &&
        kind == Lex::TokenKind::StringLiteral) {
      cpp_snippet = CppSnippet(tokens_->GetTokenText(token), indent_, style_,
                               token_infos_.Get(token).is_cpp_string);
      // An already-formatted snippet is not a rewrite: keeping the token as a
      // plain anchor keeps the edits around it minimal.
      if (cpp_snippet && *cpp_snippet != tokens_->GetTokenText(token)) {
        rewritten = *cpp_snippet;
      }
    }
    // Within a verbatim error region, the token's leading gap is the original
    // source text; otherwise the computed whitespace is recorded.
    if (std::optional<llvm::StringRef> gap = VerbatimGapBefore(token)) {
      whitespace_.AddVerbatimGapToken(gap->str(), indent_, nesting_level,
                                      token);
    } else {
      whitespace_.AddToken(newlines, spaces, indent_, nesting_level, token,
                           rewritten);
    }
    if (kind.IsOneOf({Lex::TokenKind::OpenParen,
                      Lex::TokenKind::OpenSquareBracket,
                      Lex::TokenKind::OpenCurlyBrace})) {
      ++nesting_level;
    }
    previous = token;
  }
  started_ = true;

  Lex::TokenIndex last = current_line_.back();
  last_end_byte_ =
      tokens_->GetByteOffset(last) + tokens_->GetTokenText(last).size();
  after_open_brace_ = tokens_->GetKind(last) == Lex::TokenKind::OpenCurlyBrace;
  // Record the unwrapped line's source-line extent for range expansion; see
  // `AffectedByteRanges`.
  unwrapped_line_extents_.push_back(
      {tokens_->GetLineNumber(first), tokens_->GetLineNumber(last)});
  current_line_.clear();
}

auto Formatter::Run() -> bool {
  // Format best-effort, even when the input has lex or parse errors: the parse
  // tree is always structurally valid, so output can always be produced. The
  // return value reports whether the input was error-free (so the driver can
  // reflect that in its exit code), but best-effort output is emitted either
  // way rather than giving up. The minimal error subtrees themselves are
  // emitted with their original source text (see the constructor and
  // `VerbatimGapBefore`), preserving author intent where the parse is
  // unreliable, while the surrounding code still reformats.
  //
  // TODO: `//@...` tooling directive lines (`//@include-in-dumps` and
  // `//@dump-sem-ir-begin`/`-end`) are consumed by the lexer without being
  // recorded as comments, so output reconstructed from tokens and comments
  // silently drops them. Surface them through the lexer's comment records, or
  // detect and preserve them here.
  auto comments = tokens_->comments();
  auto comment_it = comments.begin();

  for (auto token : tokens_->tokens()) {
    // Emit any comments that sort before this token. A full-line comment block
    // is re-indented to the current code indent and wrapped to the column
    // limit; a trailing comment stays on the line of the code it follows.
    while (comment_it != comments.end() &&
           tokens_->IsAfterComment(token, *comment_it)) {
      // A comment inside a verbatim error region sits in a source gap that is
      // copied verbatim, so it must not also be emitted separately.
      if (VerbatimGapBefore(token)) {
        ++comment_it;
        continue;
      }
      llvm::StringRef text = tokens_->GetCommentText(*comment_it);
      int start_byte = text.data() - tokens_->source().text().data();
      if (tokens_->IsTrailingComment(*comment_it)) {
        // Flush the code line so its tokens are recorded, then append the
        // comment to it. The comment's trailing newline is dropped, as line
        // breaks are attributed to the following content.
        FlushLine();
        whitespace_.AddTrailingComment(text.rtrim().str());
      } else {
        FlushLine();
        int leading_newlines =
            LeadingNewlines(start_byte, /*is_block_end=*/false);
        // Join the formatted comment lines with internal newlines but no
        // trailing one, and record the block as raw, verbatim text.
        whitespace_.AddRaw(leading_newlines,
                           CommentText(text, indent_, style_.column_limit));
      }
      started_ = true;
      // Comment text includes its trailing newline (though a comment ending
      // the file may lack it); exclude it from the byte baseline so a
      // following blank line is counted consistently with tokens (whose text
      // has no trailing newline).
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
        indent_ += style_.indent_width;
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
        indent_ -= style_.indent_width;
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

  output_ = whitespace_.Generate(token_map_);
  return !tokens_->has_errors() && !tree_->has_errors();
}

auto Formatter::AffectedByteRanges(LineRange lines) const
    -> llvm::SmallVector<std::pair<int32_t, int32_t>> {
  llvm::StringRef source = tokens_->source().text();

  // Work in whole source lines. The requested lines seed the set, and two
  // kinds of layout coupling expand it:
  //
  //   - A unwrapped line -- one flushed statement or line, possibly wrapped
  //     over several source lines -- lays out as a unit, so a partially
  //     affected one is wholly affected; otherwise a re-wrap could be applied
  //     in part and produce output full formatting never would.
  //   - A brace whose matching brace is affected becomes affected, so range
  //     formatting fixes a dangling brace.
  //
  // A worklist visits each newly affected line once, applying every coupling
  // that touches it; couplings chain (a brace inside a wrapped line), so this
  // reaches the fixed point in linear time.

  // Precompute the start byte of every 1-based line, plus one entry for the
  // end of the source, so line/byte conversions don't rescan the source.
  llvm::SmallVector<int32_t> line_starts;
  line_starts.push_back(0);
  for (size_t newline = source.find('\n'); newline != llvm::StringRef::npos;
       newline = source.find('\n', newline + 1)) {
    line_starts.push_back(newline + 1);
  }
  if (line_starts.back() != static_cast<int32_t>(source.size())) {
    line_starts.push_back(source.size());
  }
  int line_count = line_starts.size() - 1;

  // Collect the brace pairs. A lexer-inserted recovery token's synthesized
  // offset is not a real source position, so such a pair has no source line
  // to expand to and is skipped.
  struct BracePair {
    int open_line;
    int close_line;
  };
  llvm::SmallVector<BracePair> pairs;
  for (auto token : tokens_->tokens()) {
    if (tokens_->GetKind(token) == Lex::TokenKind::OpenCurlyBrace) {
      auto close = tokens_->GetMatchedClosingToken(token);
      if (tokens_->IsRecoveryToken(token) || tokens_->IsRecoveryToken(close)) {
        continue;
      }
      pairs.push_back({.open_line = tokens_->GetLineNumber(token),
                       .close_line = tokens_->GetLineNumber(close)});
    }
  }

  // Index the couplings by line: a non-negative id is a unwrapped-line extent,
  // and `~id` a brace pair.
  llvm::SmallVector<llvm::SmallVector<int32_t, 2>> couplings(line_count + 1);
  for (auto [i, extent] : llvm::enumerate(unwrapped_line_extents_)) {
    for (int line = extent.first; line <= extent.second; ++line) {
      couplings[line].push_back(static_cast<int32_t>(i));
    }
  }
  for (auto [i, pair] : llvm::enumerate(pairs)) {
    couplings[pair.open_line].push_back(~static_cast<int32_t>(i));
    couplings[pair.close_line].push_back(~static_cast<int32_t>(i));
  }

  llvm::BitVector affected(line_count + 1);
  llvm::SmallVector<int> worklist;
  auto add_line = [&](int line) {
    if (line >= 1 && line <= line_count && !affected[line]) {
      affected.set(line);
      worklist.push_back(line);
    }
  };
  for (int line = lines.first_line;
       line <= std::min(lines.last_line, line_count); ++line) {
    add_line(line);
  }
  while (!worklist.empty()) {
    int line = worklist.pop_back_val();
    for (int32_t id : couplings[line]) {
      if (id >= 0) {
        auto [extent_begin, extent_end] = unwrapped_line_extents_[id];
        for (int l = extent_begin; l <= extent_end; ++l) {
          add_line(l);
        }
      } else {
        const BracePair& pair = pairs[~id];
        add_line(pair.open_line);
        add_line(pair.close_line);
      }
    }
  }

  // Convert the affected lines to merged byte ranges.
  llvm::SmallVector<std::pair<int32_t, int32_t>> ranges;
  for (int line = 1; line <= line_count; ++line) {
    if (!affected[line]) {
      continue;
    }
    int32_t begin = line_starts[line - 1];
    int32_t end = line_starts[line];
    if (!ranges.empty() && ranges.back().second == begin) {
      ranges.back().second = end;
    } else {
      ranges.push_back({begin, end});
    }
  }
  return ranges;
}

auto Formatter::ComputeReplacements(std::optional<LineRange> lines) const
    -> llvm::SmallVector<Replacement> {
  llvm::StringRef source = tokens_->source().text();
  llvm::StringRef output = output_;

  // When a line range is requested, lower it to the byte ranges it affects (the
  // requested lines plus matching braces). An edit is kept when the gap it
  // rewrites lies in one of those ranges.
  llvm::SmallVector<std::pair<int32_t, int32_t>> affected;
  if (lines) {
    affected = AffectedByteRanges(*lines);
  }
  int32_t source_size = source.size();
  auto in_requested_lines = [&](int32_t begin, int32_t end) -> bool {
    if (!lines) {
      return true;
    }
    if (begin != end) {
      // A non-empty gap is in range if it overlaps an affected byte range.
      for (auto [range_begin, range_end] : affected) {
        if (begin < range_end && end > range_begin) {
          return true;
        }
      }
      return false;
    }
    // A zero-width gap is an insertion point; it is in range if it sits inside
    // an affected range, or exactly at end-of-source when an affected range
    // runs to the last line (so a missing trailing newline is still added).
    for (auto [range_begin, range_end] : affected) {
      if (begin >= range_begin &&
          (begin < range_end ||
           (begin == range_end && range_end == source_size))) {
        return true;
      }
    }
    return false;
  };

  llvm::SmallVector<Replacement> replacements;
  auto maybe_add_gap = [&](int32_t source_begin, int32_t source_end,
                           int32_t output_begin, int32_t output_end) {
    llvm::StringRef source_gap =
        source.substr(source_begin, source_end - source_begin);
    llvm::StringRef output_gap =
        output.substr(output_begin, output_end - output_begin);
    if (source_gap != output_gap &&
        in_requested_lines(source_begin, source_end)) {
      replacements.push_back({.offset = source_begin,
                              .length = source_end - source_begin,
                              .text = output_gap.str()});
    }
  };

  int32_t source_pos = 0;
  int32_t output_pos = 0;
  for (const TokenSpan& span : token_map_) {
    maybe_add_gap(source_pos, span.source_begin, output_pos, span.output_begin);
    // The token text itself is copied verbatim, so skip over it in both.
    source_pos = span.source_begin + span.length;
    output_pos = span.output_begin + span.length;
  }
  maybe_add_gap(source_pos, source.size(), output_pos, output.size());
  return replacements;
}

}  // namespace Carbon::Format
