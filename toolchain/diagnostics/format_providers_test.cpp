// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon {
namespace {

using ::testing::Eq;

TEST(FormatBool, Cases) {
  constexpr char Format[] = "{0:a|b}";
  EXPECT_THAT(llvm::formatv(Format, FormatBool{.value = true}).str(), Eq("a"));
  EXPECT_THAT(llvm::formatv(Format, FormatBool{.value = false}).str(), Eq("b"));
}

TEST(FormatBool, Spaces) {
  constexpr char Format[] = "{0: a | b }";
  EXPECT_THAT(llvm::formatv(Format, FormatBool{.value = true}).str(), Eq("a "));
  EXPECT_THAT(llvm::formatv(Format, FormatBool{.value = false}).str(),
              Eq(" b"));
}

TEST(FormatBool, QuotedSpaces) {
  constexpr char Format[] = "{0:' a | b '}";
  EXPECT_THAT(llvm::formatv(Format, FormatBool{.value = true}).str(),
              Eq(" a "));
  EXPECT_THAT(llvm::formatv(Format, FormatBool{.value = false}).str(),
              Eq(" b "));
}

TEST(FormatInt, OnlyDefault) {
  constexpr char Format[] = "{0::default}";
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 0}).str(),
              Eq("default"));
}

TEST(FormatInt, OneEquals) {
  constexpr char Format[] = "{0:=0:zero}";
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 0}).str(), Eq("zero"));
}

TEST(FormatInt, TwoEquals) {
  constexpr char Format[] = "{0:=0:zero|=1:one}";
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 0}).str(), Eq("zero"));
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 1}).str(), Eq("one"));
}

TEST(FormatInt, TwoEqualsAndDefault) {
  constexpr char Format[] = "{0:=0:zero|=1:one|:default}";
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 0}).str(), Eq("zero"));
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 1}).str(), Eq("one"));
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 2}).str(),
              Eq("default"));
}

TEST(FormatInt, Spaces) {
  constexpr char Format[] = "{0:=0: zero |=1: one |: default }";
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 0}).str(), Eq(" zero "));
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 1}).str(), Eq(" one "));
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 2}).str(),
              Eq(" default"));
}

TEST(FormatInt, QuotedSpaces) {
  constexpr char Format[] = "{0:'=0: zero |=1: one |: default '}";
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 0}).str(), Eq(" zero "));
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 1}).str(), Eq(" one "));
  EXPECT_THAT(llvm::formatv(Format, FormatInt{.value = 2}).str(),
              Eq(" default "));
}

}  // namespace
}  // namespace Carbon
