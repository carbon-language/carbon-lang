// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/whitespace_manager.h"

#include <algorithm>
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

auto WhitespaceManager::AlignTrailingComments() -> void {
  if (!style_.align_trailing_comments) {
    return;
  }
  ComputeStartColumns();

  llvm::SmallVector<int> run;
  int run_indent = -1;
  auto finalize = [&] {
    if (run.size() >= 2) {
      int target = 0;
      for (int index : run) {
        target = std::max(target, changes_[index].start_column);
      }
      for (int index : run) {
        Change& comment = changes_[index];
        comment.spaces += target - comment.start_column;
      }
    }
    run.clear();
    run_indent = -1;
  };

  int n = changes_.size();
  int i = 0;
  while (i < n) {
    int j = i + 1;
    while (j < n && changes_[j].newlines == 0) {
      ++j;
    }

    int indent = -1;
    bool has_multiline_token = false;
    int match = -1;
    for (int k = i; k < j; ++k) {
      if (changes_[k].kind == ChangeKind::Token ||
          changes_[k].kind == ChangeKind::VerbatimGapToken) {
        if (indent < 0) {
          indent = changes_[k].indent_level;
        }
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
    if (has_multiline_token || match < 0) {
      finalize();
    } else {
      if (!run.empty() && indent != run_indent) {
        finalize();
      }
      if (run.empty()) {
        run_indent = indent;
      }
      run.push_back(match);
    }
    i = j;
  }
  finalize();
}

auto WhitespaceManager::Generate() -> std::string {
  AlignTrailingComments();

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
        output.append(text.data(), text.size());
        break;
      }

      case ChangeKind::VerbatimGapToken: {
        CARBON_CHECK(!tokens_->IsRecoveryToken(change.token));
        output.append(change.gap.data(), change.gap.size());
        llvm::StringRef text = TokenText(change);
        output.append(text.data(), text.size());
        break;
      }
    }
  }
  if (!changes_.empty()) {
    output.push_back('\n');
  }
  return output;
}

}  // namespace Carbon::Format
