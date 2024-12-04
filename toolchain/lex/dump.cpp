// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/dump.h"

#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Lex::DumpOverloads {

auto Dump(TokenKind /*token_kind*/, const TokenizedBuffer& /*buffer*/) -> void {
  llvm::errs() << "TokenKind(?)\n";
}

}  // namespace Carbon::Lex::DumpOverloads
