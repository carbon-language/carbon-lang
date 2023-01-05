// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_DEFINE_RAW_ENUM_CLASS(ParserState, uint8_t) {
#define CARBON_PARSER_STATE(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/parser/parser_state.def"
};

class ParserState : public CARBON_ENUM_BASE(ParserState) {
 public:
#define CARBON_PARSER_STATE(Name) CARBON_ENUM_CONSTANT_DECLARATION(Name)
#include "toolchain/parser/parser_state.def"
};

#define CARBON_PARSER_STATE(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(ParserState, Name)
#include "toolchain/parser/parser_state.def"

// We expect ParserState to fit compactly into 8 bits.
static_assert(sizeof(ParserState) == 1, "ParserState includes padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
