// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/dump_id.h"

#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Lex {

auto DumpIdImpl(TokenIndex token, const TokenizedBuffer& buffer) -> void {
  if (!token.is_valid()) {
    llvm::errs() << "TokenIndex(invalid)";
    return;
  }

  auto kind = buffer.GetKind(token);
  auto line = buffer.GetLineNumber(token);
  auto col = buffer.GetColumnNumber(token);

  llvm::errs() << "TokenIndex(kind: ";
  kind.Print(llvm::errs());
  llvm::errs() << ", loc: ";
  llvm::errs().write_escaped(buffer.source().filename());
  llvm::errs() << ":" << line << ":" << col << ")";
}

template <>
auto DumpIdMethods<TokenizedBuffer>::Newline() const -> void {
  llvm::errs() << "\n";
}

}  // namespace Carbon::Lex
