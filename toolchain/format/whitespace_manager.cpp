// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/whitespace_manager.h"

#include <cstdint>
#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"

namespace Carbon::Format {

auto WhitespaceManager::AddToken(int newlines, int spaces,
                                 Lex::TokenIndex token) -> void {
  changes_.push_back({.is_token = true,
                      .newlines = newlines,
                      .spaces = spaces,
                      .token = token});
}

auto WhitespaceManager::AddRaw(int newlines, std::string text) -> void {
  changes_.push_back({.is_token = false,
                      .newlines = newlines,
                      .spaces = 0,
                      .raw = std::move(text)});
}

auto WhitespaceManager::AddTrailingComment(std::string text) -> void {
  // No line break: the comment appends to the current line, separated by a
  // single space.
  changes_.push_back({.is_token = false,
                      .is_trailing_comment = true,
                      .newlines = 0,
                      .spaces = 1,
                      .raw = std::move(text)});
}

auto WhitespaceManager::Generate(llvm::SmallVectorImpl<TokenSpan>& token_map)
    -> std::string {
  std::string output;
  for (const Change& change : changes_) {
    output.append(change.newlines, '\n');
    if (!change.is_token) {
      // A trailing comment appends to the current line after its separating
      // space; a full-line block is emitted verbatim.
      if (change.is_trailing_comment) {
        output.append(change.spaces, ' ');
      }
      output.append(change.raw);
      continue;
    }
    output.append(change.spaces, ' ');
    llvm::StringRef text = tokens_->GetTokenText(change.token);
    // A lexer-inserted recovery token's text does not exist in the source (its
    // byte offset is synthesized, and can even overlap a neighboring token),
    // so it cannot anchor a minimal edit. Skipping its span leaves the emitted
    // text inside the surrounding gap, where the gap's edit inserts it.
    if (!tokens_->IsRecoveryToken(change.token)) {
      token_map.push_back({.source_begin = tokens_->GetByteOffset(change.token),
                           .output_begin = static_cast<int32_t>(output.size()),
                           .length = static_cast<int32_t>(text.size())});
    }
    output.append(text.data(), text.size());
  }
  // Content never carries its own trailing newline (line breaks are attributed
  // to the following content), so a non-empty file ends with one final newline.
  if (!changes_.empty()) {
    output.push_back('\n');
  }
  return output;
}

}  // namespace Carbon::Format
