// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_STATE_H_
#define CARBON_TOOLCHAIN_PARSE_STATE_H_

#include <cstdint>

#include "common/enum_base.h"

namespace Carbon::Parse {

CARBON_DEFINE_RAW_ENUM_CLASS(StateKind, uint8_t) {
#define CARBON_PARSE_STATE(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/parse/state.def"
};

class StateKind : public CARBON_ENUM_BASE(StateKind) {
 public:
#define CARBON_PARSE_STATE(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/parse/state.def"
};

#define CARBON_PARSE_STATE(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(StateKind, Name)
#include "toolchain/parse/state.def"

// We expect State to fit compactly into 8 bits.
static_assert(sizeof(StateKind) == 1, "State includes padding!");

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_STATE_H_
