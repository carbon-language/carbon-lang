// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Diagnostics {
namespace {

using ::testing::Eq;

TEST(BoolAsSelect, Cases) {
  constexpr char Format[] = "{0:a|b}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::BoolAsSelect(true)).str(),
              Eq("a"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::BoolAsSelect(false)).str(),
              Eq("b"));
}

TEST(BoolAsSelect, CasesWithNormalFormat) {
  constexpr char Format[] = "{0} {0:a|b}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::BoolAsSelect(true)).str(),
              Eq("true a"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::BoolAsSelect(false)).str(),
              Eq("false b"));
}

TEST(BoolAsSelect, Spaces) {
  constexpr char Format[] = "{0: a | b }";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::BoolAsSelect(true)).str(),
              Eq(" a "));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::BoolAsSelect(false)).str(),
              Eq(" b "));
}

TEST(IntAsSelect, OnlyDefault) {
  constexpr char Format[] = "{0::default}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq("default"));
}

TEST(IntAsSelect, OneEquals) {
  constexpr char Format[] = "{0:=0:zero}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq("zero"));
}

TEST(IntAsSelect, TwoEquals) {
  constexpr char Format[] = "{0:=0:zero|=1:one}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq("zero"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(1)).str(),
              Eq("one"));
}

TEST(IntAsSelect, TwoEqualsAndDefault) {
  constexpr char Format[] = "{0:=0:zero|=1:one|:default}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq("zero"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(1)).str(),
              Eq("one"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(2)).str(),
              Eq("default"));
}

TEST(IntAsSelect, LessAndGreater) {
  constexpr char Format[] =
      "{0:<0:negative|>10:huge|=0:zero|<=4:small|>=7:large|:moderate}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(-1)).str(),
              Eq("negative"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq("zero"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(1)).str(),
              Eq("small"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(2)).str(),
              Eq("small"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(3)).str(),
              Eq("small"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(4)).str(),
              Eq("small"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(5)).str(),
              Eq("moderate"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(6)).str(),
              Eq("moderate"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(7)).str(),
              Eq("large"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(8)).str(),
              Eq("large"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(9)).str(),
              Eq("large"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(10)).str(),
              Eq("large"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(11)).str(),
              Eq("huge"));
}

TEST(IntAsSelect, Spaces) {
  constexpr char Format[] = "{0:=0: zero |=1: one |: default }";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq(" zero "));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(1)).str(),
              Eq(" one "));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(2)).str(),
              Eq(" default "));
}

TEST(IntAsSelect, CasesWithNormalFormat) {
  constexpr char Format[] = "{0} argument{0:=1:|:s}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq("0 arguments"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(1)).str(),
              Eq("1 argument"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(2)).str(),
              Eq("2 arguments"));
}

TEST(IntAsSelect, PluralS) {
  constexpr char Format[] = "{0} argument{0:s}";
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(0)).str(),
              Eq("0 arguments"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(1)).str(),
              Eq("1 argument"));
  EXPECT_THAT(llvm::formatv(Format, Diagnostics::IntAsSelect(2)).str(),
              Eq("2 arguments"));
}

}  // namespace
}  // namespace Carbon::Diagnostics
