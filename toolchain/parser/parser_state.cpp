// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_state.h"

#include "llvm/ADT/StringRef.h"

namespace Carbon {

auto ParserState::name() const -> llvm::StringRef {
  static constexpr llvm::StringLiteral Names[] = {
#define CARBON_PARSER_STATE(Name) #Name,
#include "toolchain/parser/parser_state.def"
  };
  return Names[static_cast<int>(state_)];
}

}  // namespace Carbon
