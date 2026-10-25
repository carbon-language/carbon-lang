// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/formatter.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/format/line_wrapper.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Format {

static auto ReindentComment(llvm::StringRef text, int indent) -> std::string {
  llvm::SmallVector<std::string> lines;
  std::string indentation(indent, ' ');
  llvm::SmallVector<llvm::StringRef> raw_lines;
  text.split(raw_lines, '\n');
  for (llvm::StringRef raw : raw_lines) {
    llvm::StringRef line = raw.trim();
    if (!line.empty()) {
      lines.push_back(indentation + line.str());
    }
  }
  return llvm::join(lines, "\n");
}

Formatter::Formatter(const Parse::Tree* tree, const Style& style)
    : tree_(tree),
      tokens_(&tree->tokens()),
      style_(style),
      whitespace_(tokens_, style),
      token_infos_(
          TokenInfoStore::MakeWithExplicitSize(tokens_->size(), TokenInfo())) {
  for (auto token : tokens_->tokens()) {
    llvm::StringRef text = tokens_->GetTokenText(token);
    if (tokens_->GetKind(token) == Lex::TokenKind::StringLiteral) {
      if (auto escaped = EscapeTripleQuoteString(text)) {
        token_infos_.Get(token).column_width =
            static_cast<int>(llvm::StringRef(*escaped).rtrim('\r').size());
        continue;
      }
    }
    token_infos_.Get(token).column_width =
        static_cast<int>(text.split('\n').first.rtrim('\r').size());
  }

  struct SubtreeRange {
    Parse::NodeKind kind;
    Lex::TokenIndex token = Lex::TokenIndex::None;
    Lex::TokenIndex min = Lex::TokenIndex::None;
    Lex::TokenIndex max = Lex::TokenIndex::None;
    Lex::TokenIndex chain_last_member = Lex::TokenIndex::None;
    int chain_length = 0;
    bool is_error_root = false;
  };
  llvm::SmallVector<SubtreeRange> range_stack;
  int32_t next_verbatim_region_id = 0;
  auto mark_verbatim = [&](const SubtreeRange& range) {
    if (!range.min.has_value()) {
      return;
    }
    int32_t region_id = next_verbatim_region_id++;
    for (int index = range.min.index; index <= range.max.index; ++index) {
      Lex::TokenIndex tok(index);
      if (!tokens_->IsRecoveryToken(tok) &&
          !tokens_->GetKind(tok).IsOneOf(
              {Lex::TokenKind::FileStart, Lex::TokenKind::FileEnd})) {
        token_infos_.Get(tok).verbatim_region_id = region_id;
      }
    }
  };
  for (auto node_id : tree_->postorder()) {
    auto kind = tree_->node_kind(node_id);
    Lex::TokenIndex token = tree_->node_token(node_id);
    TokenRole role = RoleForNodeKind(kind);
    if (role != TokenRole::Unknown && token.has_value()) {
      token_infos_.Get(token).role = role;
    }

    bool is_struct = kind == Parse::NodeKind::StructLiteral ||
                     kind == Parse::NodeKind::StructTypeLiteral ||
                     kind == Parse::NodeKind::StructPattern;
    llvm::SmallVector<Lex::TokenIndex, 8> field_commas;

    SubtreeRange range = {.kind = kind, .token = token};
    if (token.has_value() && !tokens_->IsRecoveryToken(token) &&
        !tokens_->GetKind(token).IsOneOf(
            {Lex::TokenKind::FileStart, Lex::TokenKind::FileEnd})) {
      range.min = range.max = token;
    }
    bool node_has_error = tree_->node_has_error(node_id);
    auto fold_child = [&]() -> SubtreeRange {
      CARBON_CHECK(!range_stack.empty(), "NodeId {0} ({1}) is missing children",
                   node_id, kind);
      SubtreeRange child = range_stack.pop_back_val();
      if (is_struct &&
          (child.kind == Parse::NodeKind::StructLiteralComma ||
           child.kind == Parse::NodeKind::StructTypeLiteralComma ||
           child.kind == Parse::NodeKind::PatternListComma) &&
          child.token.has_value() && !tokens_->IsRecoveryToken(child.token)) {
        field_commas.push_back(child.token);
      }
      if (child.min.has_value() &&
          (!range.min.has_value() || child.min < range.min)) {
        range.min = child.min;
      }
      if (child.max.has_value() &&
          (!range.max.has_value() || child.max > range.max)) {
        range.max = child.max;
      }
      if (child.is_error_root && !node_has_error) {
        mark_verbatim(child);
      }
      return child;
    };
    std::optional<SubtreeRange> first_child;
    if (kind.has_child_count()) {
      for (int i = 0; i < kind.child_count(); ++i) {
        first_child = fold_child();
      }
    } else {
      do {
        first_child = fold_child();
      } while (first_child->kind != kind.bracket());
    }
    range.is_error_root = node_has_error;

    if ((kind == Parse::NodeKind::CallExprStart ||
         kind == Parse::NodeKind::IndexExprStart) &&
        first_child && first_child->chain_last_member.has_value()) {
      token_infos_.Get(first_child->chain_last_member).is_call_link = true;
      range.chain_last_member = first_child->chain_last_member;
      range.chain_length = first_child->chain_length;
    } else if ((kind == Parse::NodeKind::CallExpr ||
                kind == Parse::NodeKind::IndexExpr) &&
               first_child) {
      range.chain_last_member = first_child->chain_last_member;
      range.chain_length = first_child->chain_length;
    }

    int member_penalty = MemberAccessBreakPenalty(kind);
    if (member_penalty >= 0 && token.has_value() &&
        !tokens_->IsRecoveryToken(token) && range.min.has_value()) {
      if (first_child && first_child->chain_last_member.has_value()) {
        if (first_child->chain_length >= 2) {
          token_infos_.Get(first_child->chain_last_member)
              .break_penalty_before = member_penalty;
        }
        range.chain_length = first_child->chain_length + 1;
      } else {
        range.chain_length = 1;
      }
      range.chain_last_member = token;

      TokenInfo& info = token_infos_.Get(token);
      info.member_chain_id = range.min.index;
      info.break_penalty_before = range.chain_length >= 2
                                      ? MemberAccessLastLinkBreakPenalty
                                      : member_penalty;
    }

    if (is_struct && first_child && first_child->token.has_value() &&
        !tokens_->IsRecoveryToken(first_child->token) &&
        tokens_->GetKind(first_child->token) ==
            Lex::TokenKind::OpenCurlyBrace) {
      Lex::TokenIndex open = first_child->token;
      Lex::TokenIndex close = tokens_->GetMatchedClosingToken(open);
      bool is_unclosed =
          !close.has_value() || tokens_->IsRecoveryToken(close) ||
          tokens_->GetKind(close) != Lex::TokenKind::CloseCurlyBrace;
      bool has_trailing_comma = !is_unclosed && !field_commas.empty() &&
                                NextRealToken(field_commas.front()) == close;
      bool is_empty_with_comment = false;
      if (!is_unclosed && NextRealToken(open) == close) {
        auto comments = tokens_->comments();
        auto it = std::partition_point(
            comments.begin(), comments.end(), [&](Lex::CommentIndex c) {
              return tokens_->IsAfterComment(open, c);
            });
        is_empty_with_comment =
            it != comments.end() && tokens_->IsAfterComment(close, *it);
      }
      if (is_unclosed || has_trailing_comma || is_empty_with_comment) {
        token_infos_.Get(open).role = TokenRole::MultilineStructOpen;
        if (has_trailing_comma) {
          for (Lex::TokenIndex comma : field_commas) {
            token_infos_.Get(comma).role = TokenRole::BlockComma;
          }
        }
      } else {
        token_infos_.Get(open).role = TokenRole::StructOpen;
        token_infos_.Get(close).role = TokenRole::StructClose;
      }
    }

    auto op_info = OperatorInfoForNodeKind(kind, style_);
    if (op_info.break_penalty >= 0 && token.has_value() &&
        !tokens_->IsRecoveryToken(token)) {
      token_infos_.Get(token).break_penalty_after = op_info.break_penalty;
      if (op_info.aligns_operands && range.min.has_value()) {
        ++token_infos_.Get(range.min).open_scopes;
        ++token_infos_.Get(range.max).close_scopes;
      }
    } else if (kind == Parse::NodeKind::WhereExpr && token.has_value() &&
               !tokens_->IsRecoveryToken(token) && range.max.has_value()) {
      Lex::TokenIndex first_req = NextRealToken(token);
      if (first_req <= range.max) {
        ++token_infos_.Get(first_req).open_scopes;
        ++token_infos_.Get(range.max).close_scopes;
      }
    }
    range_stack.push_back(range);
  }
  for (const SubtreeRange& range : range_stack) {
    if (range.is_error_root) {
      mark_verbatim(range);
    }
  }
}

auto Formatter::NextRealToken(Lex::TokenIndex token) const -> Lex::TokenIndex {
  Lex::TokenIndex next = NextToken(token);
  while (tokens_->IsRecoveryToken(next)) {
    next = NextToken(next);
  }
  return next;
}

auto Formatter::ClampBlankLines(llvm::StringRef gap) const -> int {
  return std::clamp(static_cast<int>(gap.count('\n')) - 1, 0,
                    style_.max_empty_lines_to_keep);
}

auto Formatter::IsAttachedTrailingComment(llvm::StringRef source, int prev_end,
                                          int start_byte,
                                          Lex::CommentIndex comment) const
    -> bool {
  return tokens_->IsTrailingComment(comment) && start_byte >= prev_end &&
         source.slice(prev_end, start_byte).trim(" \t").empty();
}

auto Formatter::ComputeBlankLines(int next_start_byte, bool is_block_end)
    -> int {
  if (!last_end_byte_ || after_open_brace_ || is_block_end ||
      next_start_byte <= *last_end_byte_) {
    return 0;
  }
  return ClampBlankLines(
      tokens_->source().text().slice(*last_end_byte_, next_start_byte));
}

auto Formatter::VerbatimGapBefore(Lex::TokenIndex token) const
    -> std::optional<llvm::StringRef> {
  int32_t region_id = token_infos_.Get(token).verbatim_region_id;
  if (region_id < 0 || token.index == 0 || tokens_->IsRecoveryToken(token)) {
    return std::nullopt;
  }
  Lex::TokenIndex prev(token.index - 1);
  if (token_infos_.Get(prev).verbatim_region_id != region_id ||
      tokens_->IsRecoveryToken(prev)) {
    return std::nullopt;
  }
  int32_t prev_end = tokens_->GetByteOffset(prev) +
                     static_cast<int32_t>(tokens_->GetTokenText(prev).size());
  int32_t start = tokens_->GetByteOffset(token);
  if (start < prev_end) {
    return std::nullopt;
  }
  llvm::StringRef gap = tokens_->source().text().slice(prev_end, start);
  if (gap.empty() && tokens_->GetKind(prev) == Lex::TokenKind::StringLiteral &&
      tokens_->GetTokenText(prev) == "\"\"" &&
      tokens_->GetTokenText(token).starts_with('"')) {
    return std::nullopt;
  }
  return gap;
}

auto Formatter::FlushLine() -> void {
  if (current_line_.empty()) {
    return;
  }

  Lex::TokenIndex first = current_line_.front();
  int leading_newlines = LeadingNewlines(
      tokens_->GetByteOffset(first),
      tokens_->GetKind(first) == Lex::TokenKind::CloseCurlyBrace);

  llvm::SmallVector<int> newline_indents;
  bool has_verbatim = llvm::any_of(current_line_, [&](Lex::TokenIndex token) {
    return token_infos_.Get(token).verbatim_region_id >= 0;
  });
  if (has_verbatim ||
      indent_ + RenderedWidth(*tokens_, token_infos_, current_line_) <=
          style_.column_limit) {
    newline_indents.assign(current_line_.size(), -1);
  } else {
    newline_indents =
        SolveLineBreaks(*tokens_, token_infos_, current_line_, indent_, style_);
  }

  std::optional<Lex::TokenIndex> previous;
  for (int i = 0; i < static_cast<int>(current_line_.size()); ++i) {
    Lex::TokenIndex token = current_line_[i];
    int newlines = 0;
    int spaces = 0;
    if (!previous) {
      newlines = leading_newlines;
      spaces = indent_;
    } else if (newline_indents[i] >= 0) {
      newlines = 1;
      spaces = newline_indents[i];
    } else {
      spaces = SpacesBefore(*tokens_, token_infos_, *previous, token);
    }
    std::string rewritten;
    if (tokens_->GetKind(token) == Lex::TokenKind::StringLiteral) {
      if (auto escaped =
              EscapeTripleQuoteString(tokens_->GetTokenText(token))) {
        rewritten = std::move(*escaped);
      }
    }
    if (std::optional<llvm::StringRef> gap = VerbatimGapBefore(token)) {
      whitespace_.AddVerbatimGapToken(*gap, indent_, token,
                                      std::move(rewritten));
    } else {
      whitespace_.AddToken(newlines, spaces, indent_, token,
                           std::move(rewritten));
    }
    previous = token;
  }
  started_ = true;

  Lex::TokenIndex last = current_line_.back();
  last_end_byte_ =
      tokens_->GetByteOffset(last) + tokens_->GetTokenText(last).size();
  after_open_brace_ =
      tokens_->GetKind(last) == Lex::TokenKind::OpenCurlyBrace &&
      token_infos_.Get(last).role != TokenRole::StructOpen;
  current_line_.clear();
}

auto Formatter::Run() -> bool {
  llvm::StringRef source = tokens_->source().text();
  auto comments = tokens_->comments();
  auto comment_it = comments.begin();

  for (auto token : tokens_->tokens()) {
    if (comment_it != comments.end() &&
        tokens_->IsAfterComment(token, *comment_it)) {
      if (VerbatimGapBefore(token).has_value()) {
        while (comment_it != comments.end() &&
               tokens_->IsAfterComment(token, *comment_it)) {
          ++comment_it;
        }
      } else {
        FlushLine();
        while (comment_it != comments.end() &&
               tokens_->IsAfterComment(token, *comment_it)) {
          llvm::StringRef text = tokens_->GetCommentText(*comment_it);
          int start_byte = text.data() - source.data();
          llvm::StringRef trimmed = text.rtrim();
          if (last_end_byte_ &&
              IsAttachedTrailingComment(source, *last_end_byte_, start_byte,
                                        *comment_it)) {
            whitespace_.AddTrailingComment(trimmed);
          } else {
            int leading_newlines =
                LeadingNewlines(start_byte, /*is_block_end=*/false);
            whitespace_.AddRaw(leading_newlines,
                               ReindentComment(text, indent_));
            after_open_brace_ = false;
          }
          started_ = true;
          last_end_byte_ = start_byte + trimmed.size();
          ++comment_it;
        }
      }
    }

    if (tokens_->IsRecoveryToken(token)) {
      switch (tokens_->GetKind(token)) {
        case Lex::TokenKind::OpenCurlyBrace:
          if (!current_line_.empty() &&
              NextToken(token) != tokens_->GetMatchedClosingToken(token)) {
            FlushLine();
          }
          indent_ += style_.indent_width;
          after_open_brace_ = true;
          break;
        case Lex::TokenKind::CloseCurlyBrace:
          FlushLine();
          indent_ = std::max(0, indent_ - style_.indent_width);
          after_open_brace_ = false;
          break;
        default:
          break;
      }
      continue;
    }

    switch (tokens_->GetKind(token)) {
      case Lex::TokenKind::FileStart:
        break;

      case Lex::TokenKind::FileEnd:
        FlushLine();
        break;

      case Lex::TokenKind::OpenCurlyBrace: {
        if (token_infos_.Get(token).role == TokenRole::StructOpen) {
          current_line_.push_back(token);
          break;
        }
        current_line_.push_back(token);
        auto close = tokens_->GetMatchedClosingToken(token);
        bool is_unclosed = tokens_->IsRecoveryToken(close);
        bool has_inner_token = NextToken(token) != close;
        bool has_inner_comment =
            comment_it != comments.end() &&
            (is_unclosed ||
             tokens_->GetCommentText(*comment_it).data() - source.data() <
                 tokens_->GetByteOffset(close));
        if (is_unclosed || has_inner_token || has_inner_comment) {
          FlushLine();
        }
        indent_ += style_.indent_width;
        break;
      }

      case Lex::TokenKind::CloseCurlyBrace: {
        if (token_infos_.Get(token).role == TokenRole::StructClose) {
          current_line_.push_back(token);
          break;
        }
        auto open = tokens_->GetMatchedOpeningToken(token);
        if (!current_line_.empty() && current_line_.back() != open) {
          FlushLine();
        }
        indent_ = std::max(0, indent_ - style_.indent_width);
        current_line_.push_back(token);
        bool is_empty = !tokens_->IsRecoveryToken(open) &&
                        current_line_.size() >= 2 &&
                        current_line_[current_line_.size() - 2] == open;
        TokenRole open_role = token_infos_.Get(open).role;
        bool is_block_close = open_role == TokenRole::BlockOpen ||
                              (open_role == TokenRole::Unknown && !is_empty);
        if (is_block_close &&
            !tokens_->GetKind(NextRealToken(token))
                 .IsOneOf({Lex::TokenKind::Semi, Lex::TokenKind::Comma,
                           Lex::TokenKind::CloseParen,
                           Lex::TokenKind::CloseSquareBracket,
                           Lex::TokenKind::Else})) {
          FlushLine();
        }
        break;
      }

      case Lex::TokenKind::Comma:
        current_line_.push_back(token);
        if (token_infos_.Get(token).role == TokenRole::BlockComma) {
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

  output_ = whitespace_.Generate();
  return !tokens_->has_errors() && !tree_->has_errors();
}

}  // namespace Carbon::Format
