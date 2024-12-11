// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/lex/dump_id.h"

#include "common/ostream.h"

namespace Carbon::Lex {

auto DumpIdImpl(const TokenizedBuffer& tokens, TokenIndex token) -> void {
  if (!token.is_valid()) {
    llvm::errs() << "TokenIndex(invalid)";
    return;
  }

  auto kind = tokens.GetKind(token);
  auto line = tokens.GetLineNumber(token);
  auto col = tokens.GetColumnNumber(token);

  llvm::errs() << "TokenIndex(kind: ";
  kind.Print(llvm::errs());
  llvm::errs() << ", loc: ";
  llvm::errs().write_escaped(tokens.source().filename());
  llvm::errs() << ":" << line << ":" << col << ")";
}

LLVM_DUMP_METHOD auto DumpId(const Lex::TokenizedBuffer& tokens,
                             Lex::TokenIndex token) -> void {
  DumpIdImpl(tokens, token);
  llvm::errs() << '\n';
}

}  // namespace Carbon::Lex

#endif  // NDEBUG
