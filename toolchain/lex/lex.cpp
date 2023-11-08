// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/lex.h"

#include "toolchain/lex/lexer.h"

namespace Carbon::Lex {

auto Lex(SharedValueStores& value_stores, SourceBuffer& source,
         DiagnosticConsumer& consumer) -> TokenizedBuffer {
  return Lexer(value_stores, source, consumer).Lex();
}

}  // namespace Carbon::Lex
