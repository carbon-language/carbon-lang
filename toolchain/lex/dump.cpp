// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/lex/dump.h"

#include "common/ostream.h"

namespace Carbon::Lex {

auto DumpNoNewline(const TokenizedBuffer& tokens, TokenIndex token) -> void {
  if (!token.has_value()) {
    llvm::errs() << "TokenIndex(<none>)";
    return;
  }

  auto kind = tokens.GetKind(token);
  auto line = tokens.GetLineNumber(token);
  auto col = tokens.GetColumnNumber(token);

  llvm::errs() << "TokenIndex(kind: " << kind << ", loc: ";
  llvm::errs().write_escaped(tokens.source().filename());
  llvm::errs() << ":" << line << ":" << col << ")";
}

LLVM_DUMP_METHOD auto Dump(const TokenizedBuffer& tokens, TokenIndex token)
    -> void {
  DumpNoNewline(tokens, token);
  llvm::errs() << '\n';
}

}  // namespace Carbon::Lex

#endif  // NDEBUG
