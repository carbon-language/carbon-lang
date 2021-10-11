// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parse_node_kind.h"

#include <gtest/gtest.h>

#include <cstring>

#include "llvm/ADT/StringRef.h"

namespace Carbon {

namespace {

// Not much to test here, so just verify that the API compiles and returns the
// data in the `.def` file.
#define CARBON_PARSE_NODE_KIND(Name)                   \
  TEST(ParseNodeKindTest, Name) {                      \
    EXPECT_EQ(#Name, ParseNodeKind::Name().GetName()); \
  }
#include "toolchain/parser/parse_node_kind.def"

}  // namespace
}  // namespace Carbon
