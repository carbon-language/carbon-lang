// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/whitespace_manager.h"

#include <string>
#include <utility>

#include "common/check.h"
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
  (void)style_;
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

auto WhitespaceManager::Generate() -> std::string {
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
  // Line breaks are attributed to the following item, so a non-empty file ends
  // with one final newline.
  if (!changes_.empty()) {
    output.push_back('\n');
  }
  return output;
}

}  // namespace Carbon::Format
