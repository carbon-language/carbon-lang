// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/lex/dump.h"

#include "common/raw_string_ostream.h"

namespace Carbon::Lex {

LLVM_DUMP_METHOD auto Dump(const TokenizedBuffer& tokens, TokenIndex token)
    -> std::string {
  RawStringOstream out;
  if (!token.has_value()) {
    out << "TokenIndex(<none>)";
    return out.TakeStr();
  }

  auto kind = tokens.GetKind(token);
  auto line = tokens.GetLineNumber(token);
  auto col = tokens.GetColumnNumber(token);

  out << "TokenIndex(kind: " << kind
      << ", loc: " << FormatEscaped(tokens.source().filename()) << ":" << line
      << ":" << col << ")";
  return out.TakeStr();
}

}  // namespace Carbon::Lex

#endif  // NDEBUG
