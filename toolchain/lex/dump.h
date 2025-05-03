// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This library contains functions to assist dumping objects to stderr during
// interactive debugging. Functions named `Dump` are intended for direct use by
// developers, and should use overload resolution to determine which will be
// invoked. The debugger should do namespace resolution automatically. For
// example:
//
// - lldb: `expr Dump(tokens, id)`
// - gdb: `call Dump(tokens, id)`
//
// The `DumpNoNewline` functions are helpers that exclude a trailing newline.
// They're intended to be composed by `Dump` function implementations.

#ifndef CARBON_TOOLCHAIN_LEX_DUMP_H_
#define CARBON_TOOLCHAIN_LEX_DUMP_H_

#ifndef NDEBUG

#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Lex {

auto Dump(const TokenizedBuffer& tokens, TokenIndex token) -> std::string;

}  // namespace Carbon::Lex

#endif  // NDEBUG

#endif  // CARBON_TOOLCHAIN_LEX_DUMP_H_
