// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/whitespace_manager.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Format {

auto WhitespaceManager::OwnText(std::string text) -> llvm::StringRef {
  owned_texts_.push_back(std::move(text));
  return owned_texts_.back();
}

auto WhitespaceManager::TokenText(const Change& change) const
    -> llvm::StringRef {
  if ((change.kind == ChangeKind::Token ||
       change.kind == ChangeKind::VerbatimGapToken) &&
      !change.text.empty()) {
    return change.text;
  }
  return tokens_->GetTokenText(change.token);
}

auto WhitespaceManager::AddToken(int newlines, int spaces, int indent_level,
                                 Lex::TokenIndex token, std::string rewritten)
    -> void {
  changes_.push_back({.kind = ChangeKind::Token,
                      .newlines = newlines,
                      .spaces = spaces,
                      .indent_level = indent_level,
                      .token = token,
                      .text = rewritten.empty()
                                  ? llvm::StringRef()
                                  : OwnText(std::move(rewritten))});
}

auto WhitespaceManager::AddVerbatimGapToken(llvm::StringRef gap,
                                            int indent_level,
                                            Lex::TokenIndex token,
                                            std::string rewritten) -> void {
  // Record `newlines` from `gap` so `AlignTrailingComments` partitions physical
  // lines accurately while `Generate` emits `gap` verbatim.
  changes_.push_back({.kind = ChangeKind::VerbatimGapToken,
                      .newlines = static_cast<int>(gap.count('\n')),
                      .spaces = 0,
                      .indent_level = indent_level,
                      .token = token,
                      .gap = gap,
                      .text = rewritten.empty()
                                  ? llvm::StringRef()
                                  : OwnText(std::move(rewritten))});
}

auto WhitespaceManager::AddRaw(int newlines, std::string text) -> void {
  changes_.push_back({.kind = ChangeKind::FullLineComment,
                      .newlines = newlines,
                      .spaces = 0,
                      .text = OwnText(std::move(text))});
}

auto WhitespaceManager::AddTrailingComment(llvm::StringRef text) -> void {
  changes_.push_back({.kind = ChangeKind::TrailingComment,
                      .newlines = 0,
                      .spaces = 1,
                      .text = text});
}

auto WhitespaceManager::ComputeStartColumns() -> void {
  int column = 0;
  for (Change& change : changes_) {
    switch (change.kind) {
      case ChangeKind::FullLineComment:
        // Full-line comments end at a line boundary; the following item starts
        // a new line.
        column = 0;
        break;

      case ChangeKind::TrailingComment:
        column += change.spaces;
        change.start_column = column;
        column += static_cast<int>(change.text.rtrim('\r').size());
        break;

      case ChangeKind::Token:
      case ChangeKind::VerbatimGapToken: {
        if (change.kind == ChangeKind::VerbatimGapToken) {
          size_t last_break = change.gap.rfind('\n');
          llvm::StringRef gap_tail = last_break == llvm::StringRef::npos
                                         ? change.gap
                                         : change.gap.substr(last_break + 1);
          column = (last_break == llvm::StringRef::npos ? column : 0) +
                   static_cast<int>(gap_tail.rtrim('\r').size());
        } else {
          column = (change.newlines > 0 ? 0 : column) + change.spaces;
        }
        change.start_column = column;
        llvm::StringRef text = TokenText(change);
        // Multi-line tokens (such as block string literals) end on their last
        // physical line, whose width determines the column after the token.
        size_t last_line = text.rfind('\n');
        llvm::StringRef tail = last_line == llvm::StringRef::npos
                                   ? text
                                   : text.substr(last_line + 1);
        column = (last_line == llvm::StringRef::npos ? column : 0) +
                 static_cast<int>(tail.rtrim('\r').size());
        break;
      }
    }
  }
}

auto WhitespaceManager::AlignTrailingComments(
    llvm::SmallVectorImpl<std::pair<int, int>>& aligned_line_extents) -> void {
  if (!style_.align_trailing_comments) {
    return;
  }
  ComputeStartColumns();

  struct RunEntry {
    int change_index;
    int first_line;
    int last_line;
  };
  llvm::SmallVector<RunEntry> run;
  int run_indent = -1;
  auto finalize = [&] {
    if (run.size() >= 2) {
      int target = 0;
      int first_line = 0;
      int last_line = 0;
      for (const RunEntry& entry : run) {
        target = std::max(target, changes_[entry.change_index].start_column);
        if (entry.first_line > 0) {
          first_line = first_line > 0 ? std::min(first_line, entry.first_line)
                                      : entry.first_line;
          last_line = std::max(last_line, entry.last_line);
        }
      }
      for (const RunEntry& entry : run) {
        Change& comment = changes_[entry.change_index];
        comment.spaces += target - comment.start_column;
      }
      if (first_line > 0) {
        aligned_line_extents.push_back({first_line, last_line});
      }
    }
    run.clear();
    run_indent = -1;
  };

  int n = changes_.size();
  int i = 0;
  while (i < n) {
    // Partition `[i, j)` into one logical output line (its leading item plus
    // same-line tokens and any trailing comment).
    int j = i + 1;
    while (j < n && changes_[j].newlines == 0) {
      ++j;
    }

    int indent = -1;
    int line_first = 0;
    int line_last = 0;
    bool has_multiline_token = false;
    int match = -1;
    for (int k = i; k < j; ++k) {
      if (changes_[k].kind == ChangeKind::Token ||
          changes_[k].kind == ChangeKind::VerbatimGapToken) {
        if (indent < 0) {
          indent = changes_[k].indent_level;
        }
        int tok_line = tokens_->GetLineNumber(changes_[k].token);
        line_first = line_first > 0 ? std::min(line_first, tok_line) : tok_line;
        line_last = std::max(line_last, tok_line);
        if (TokenText(changes_[k]).contains('\n')) {
          has_multiline_token = true;
        }
      } else if (changes_[k].kind == ChangeKind::TrailingComment) {
        match = k;
      }
    }

    if (changes_[i].newlines > 1) {
      finalize();
    }
    // Multi-line tokens (such as block string literals) span physical lines of
    // body text without trailing comments, so they break any surrounding run.
    if (has_multiline_token || match < 0) {
      finalize();
    } else {
      if (!run.empty() && indent != run_indent) {
        finalize();
      }
      if (run.empty()) {
        run_indent = indent;
      }
      run.push_back({.change_index = match,
                     .first_line = line_first,
                     .last_line = line_last});
    }
    i = j;
  }
  finalize();
}

auto WhitespaceManager::Generate(
    llvm::SmallVectorImpl<TokenSpan>& token_map,
    llvm::SmallVectorImpl<std::pair<int, int>>& aligned_line_extents)
    -> std::string {
  AlignTrailingComments(aligned_line_extents);

  std::string output;
  for (const Change& change : changes_) {
    switch (change.kind) {
      case ChangeKind::FullLineComment:
        output.append(change.newlines, '\n');
        output.append(change.text.data(), change.text.size());
        break;

      case ChangeKind::TrailingComment:
        output.append(change.newlines, '\n');
        output.append(change.spaces, ' ');
        output.append(change.text.data(), change.text.size());
        break;

      case ChangeKind::Token: {
        CARBON_CHECK(!tokens_->IsRecoveryToken(change.token));
        output.append(change.newlines, '\n');
        output.append(change.spaces, ' ');
        llvm::StringRef text = TokenText(change);
        // Rewritten tokens (such as reformatted C++ snippets) omit their
        // `TokenSpan` anchor so the rewrite folds into the surrounding gap.
        if (change.text.empty()) {
          token_map.push_back(
              {.source_begin = tokens_->GetByteOffset(change.token),
               .output_begin = static_cast<int32_t>(output.size()),
               .length = static_cast<int32_t>(text.size())});
        }
        output.append(text.data(), text.size());
        break;
      }

      case ChangeKind::VerbatimGapToken: {
        CARBON_CHECK(!tokens_->IsRecoveryToken(change.token));
        output.append(change.gap.data(), change.gap.size());
        llvm::StringRef text = TokenText(change);
        if (change.text.empty()) {
          token_map.push_back(
              {.source_begin = tokens_->GetByteOffset(change.token),
               .output_begin = static_cast<int32_t>(output.size()),
               .length = static_cast<int32_t>(text.size())});
        }
        output.append(text.data(), text.size());
        break;
      }
    }
  }
  // Line breaks are attributed to the following item, so a non-empty file ends
  // with one final newline.
  if (!changes_.empty()) {
    output.push_back('\n');
  }
  return output;
}

}  // namespace Carbon::Format
