// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

CARBON_ENUM_BASE_1_OF_7(ParserStateBase)
#define CARBON_PARSER_STATE(Name) CARBON_ENUM_BASE_2_OF_7_ITER(Name)
#include "toolchain/parser/parser_state.def"
CARBON_ENUM_BASE_3_OF_7(ParserStateBase)
#define CARBON_PARSER_STATE(Name) CARBON_ENUM_BASE_4_OF_7_ITER(Name)
#include "toolchain/parser/parser_state.def"
CARBON_ENUM_BASE_5_OF_7(ParserStateBase)
#define CARBON_PARSER_STATE(Name) CARBON_ENUM_BASE_6_OF_7_ITER(Name)
#include "toolchain/parser/parser_state.def"
CARBON_ENUM_BASE_7_OF_7(ParserStateBase)

class ParserState : public ParserStateBase<ParserState> {
  using ParserStateBase::ParserStateBase;
};

// We expect ParserState to fit compactly into 8 bits.
static_assert(sizeof(ParserState) == 1, "ParserState includes padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
