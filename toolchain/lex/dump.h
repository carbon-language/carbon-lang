// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_DUMP_H_
#define CARBON_TOOLCHAIN_LEX_DUMP_H_

#include "toolchain/lex/token_kind.h"

namespace Carbon::Lex {

class TokenizedBuffer;

namespace DumpOverloads {

auto Dump(TokenKind token_kind, const TokenizedBuffer& buffer) -> void;

}

// A set of Dump() overloads that dump an object to stderr, useful for calling
// inside a debugger. These are all exposed as part of the
// `Lex::TokenizedBuffer` API.
//
// This class is inherited by `Lex::TokenizedBuffer`, which provides itself as
// the template parameter.
template <class TokenizedBuffer>
class DumpMethods {
  static_assert(std::same_as<TokenizedBuffer, ::Carbon::Lex::TokenizedBuffer>);

 public:
#define CARBON_LEX_DUMP_TYPE(Type)                                      \
  LLVM_DUMP_METHOD auto Dump(Type t) -> void {                          \
    DumpOverloads::Dump(t, static_cast<const TokenizedBuffer&>(*this)); \
  }
#include "toolchain/lex/dump.def"
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_DUMP_H_
