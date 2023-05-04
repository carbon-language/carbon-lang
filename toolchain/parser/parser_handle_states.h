// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_HANDLE_STATES_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_HANDLE_STATES_H_

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// `clang-format` has a bug with spacing around `->` returns in macros. See
// https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_PARSER_STATE(Name) \
  auto ParserHandle##Name(ParserContext& context)->void;
#include "toolchain/parser/parser_state.def"

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_HANDLE_STATES_H_
