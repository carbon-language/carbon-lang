// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/format.h"

#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

// TODO: Add support for formatting line ranges (will need flags too).
auto Format(const Lex::TokenizedBuffer& tokens, llvm::raw_ostream& out)
    -> bool {
  if (tokens.has_errors()) {
    // TODO: Error recovery.
    return false;
  }

  auto comments = tokens.comments();
  auto comment_it = comments.begin();

  llvm::ListSeparator sep(" ");

  for (auto token : tokens.tokens()) {
    while (comment_it != comments.end() &&
           tokens.IsAfterComment(token, *comment_it)) {
      // TODO: Fix newlines and indent.
      out << "\n" << tokens.GetCommentText(*comment_it) << "\n";
      ++comment_it;
    }

    switch (tokens.GetKind(token)) {
      case Lex::TokenKind::FileStart:
        break;
      case Lex::TokenKind::FileEnd:
        out << "\n";
        break;
      default:
        // TODO: More dependent formatting.
        out << sep << tokens.GetTokenText(token);
        break;
    }
  }
  return true;
}

}  // namespace Carbon::Format
