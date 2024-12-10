// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_DUMP_ID_H_
#define CARBON_TOOLCHAIN_LEX_DUMP_ID_H_

#include "toolchain/lex/token_index.h"

namespace Carbon::Lex {

class TokenizedBuffer;

auto DumpIdImpl(const TokenizedBuffer& buffer, TokenIndex token) -> void;

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_DUMP_ID_H_
