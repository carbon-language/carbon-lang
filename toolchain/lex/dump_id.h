// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_DUMP_ID_H_
#define CARBON_TOOLCHAIN_LEX_DUMP_ID_H_

#include "toolchain/lex/token_index.h"

namespace Carbon::Lex {

class TokenizedBuffer;

namespace DumpIdOverloads {

auto DumpId(TokenIndex token, const TokenizedBuffer& buffer) -> void;

}  // namespace DumpIdOverloads

// A set of DumpId() overloads that dump an object to stderr, useful for calling
// inside a debugger. These are all exposed as part of the
// `Lex::TokenizedBuffer` API.
//
// This class is inherited by `Lex::TokenizedBuffer`, which provides itself as
// the template parameter.
template <class TokenizedBuffer>
class DumpIdMethods {
  static_assert(std::same_as<TokenizedBuffer, ::Carbon::Lex::TokenizedBuffer>);

 public:
  LLVM_DUMP_METHOD auto DumpId(TokenIndex token) const -> void {
    DumpIdOverloads::DumpId(token, static_cast<const TokenizedBuffer&>(*this));
    Newline();
  }

 private:
  auto Newline() const -> void;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_DUMP_ID_H_
