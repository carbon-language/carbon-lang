// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_DUMP_ID_H_
#define CARBON_TOOLCHAIN_LEX_DUMP_ID_H_

#ifndef NDEBUG

#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Lex {

class TokenizedBuffer;

auto DumpIdImpl(const TokenizedBuffer& tokens, TokenIndex token) -> void;

// A set of DumpId() overloads that dump an object to stderr, useful for
// calling inside a debugger.
auto DumpId(const Lex::TokenizedBuffer& tokens, Lex::TokenIndex token) -> void;

}  // namespace Carbon::Lex

#endif  // NDEBUG

#endif  // CARBON_TOOLCHAIN_LEX_DUMP_ID_H_
