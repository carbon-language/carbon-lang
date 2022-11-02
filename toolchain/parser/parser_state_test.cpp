// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_state.h"

#include <gtest/gtest.h>

#include <cstring>

#include "llvm/ADT/StringRef.h"

namespace Carbon::Testing {
namespace {

// Not much to test here, so just verify that the API compiles and returns the
// data in the `.def` file.
#define CARBON_PARSER_STATE(Name) \
  TEST(ParserState, Name) { EXPECT_EQ(#Name, ParserState::Name().name()); }
#include "toolchain/parser/parser_state.def"

}  // namespace
}  // namespace Carbon::Testing
