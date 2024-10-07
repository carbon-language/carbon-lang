// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/format.h"

#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Format {

auto Format(const Lex::TokenizedBuffer& tokens, llvm::raw_ostream& out)
    -> bool {
  if (tokens.has_errors()) {
    // TODO: Error recovery.
    return false;
  }
  llvm::ListSeparator sep(" ");
  for (auto token : tokens.tokens()) {
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
