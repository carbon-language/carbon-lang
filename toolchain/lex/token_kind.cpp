// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"

namespace Carbon::Lex {

CARBON_DEFINE_ENUM_CLASS_NAMES(TokenKind) {
#define CARBON_TOKEN(TokenName) CARBON_ENUM_CLASS_NAME_STRING(TokenName)
#include "toolchain/lex/token_kind.def"
};

}  // namespace Carbon::Lex
