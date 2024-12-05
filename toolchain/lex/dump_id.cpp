// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/dump_id.h"

#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Lex::DumpIdOverloads {

auto DumpId(TokenIndex /*token*/, const TokenizedBuffer& /*buffer*/) -> void {
  llvm::errs() << "TokenIndex(?)";
}

}  // namespace Carbon::Lex::DumpIdOverloads

namespace Carbon::Lex {
template <>
auto DumpIdMethods<TokenizedBuffer>::Newline() const -> void {
  llvm::errs() << "\n";
}
}  // namespace Carbon::Lex
