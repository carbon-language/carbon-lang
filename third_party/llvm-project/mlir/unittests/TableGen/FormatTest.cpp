//===- FormatTest.cpp - TableGen Format Utility Tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Format.h"
#include "gmock/gmock.h"

using mlir::tblgen::FmtContext;
using mlir::tblgen::tgfmt;
using ::testing::StrEq;

TEST(FormatTest, EmptyFmtStr) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("", &ctx));
  EXPECT_TRUE(result.empty());
}

/// Allow extra unused positional parameters.
TEST(FormatTest, EmptyFmtStrExtraParams) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("", &ctx, "a", "b", "c"));
  EXPECT_TRUE(result.empty());
}

/// Allow unused placeholder substitution in context.
TEST(FormatTest, EmptyFmtStrPopulatedCtx) {
  FmtContext ctx;
  ctx.withBuilder("builder");
  std::string result = std::string(tgfmt("", &ctx));
  EXPECT_TRUE(result.empty());
}

TEST(FormatTest, LiteralFmtStr) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("void foo {}", &ctx));
  EXPECT_THAT(result, StrEq("void foo {}"));
}

/// Print single dollar literally.
TEST(FormatTest, AdjacentDollar) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$", &ctx));
  EXPECT_THAT(result, StrEq("$"));
}

/// Print dangling dollar literally.
TEST(FormatTest, DanglingDollar) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("foo bar baz$", &ctx));
  EXPECT_THAT(result, StrEq("foo bar baz$"));
}

/// Allow escape dollars with '$$'.
TEST(FormatTest, EscapeDollars) {
  FmtContext ctx;
  std::string result =
      std::string(tgfmt("$$ $$$$ $$$0 $$$_self", &ctx.withSelf("self"), "-0"));
  EXPECT_THAT(result, StrEq("$ $$ $-0 $self"));
}

TEST(FormatTest, PositionalFmtStr) {
  FmtContext ctx;
  std::string b = "b";
  int c = 42;
  char d = 'd';
  std::string result =
      std::string(tgfmt("$0 $1 $2 $3", &ctx, "a", b, c + 1, d));
  EXPECT_THAT(result, StrEq("a b 43 d"));
}

/// Output the placeholder if missing substitution.
TEST(FormatTest, PositionalFmtStrMissingParams) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$0 %1 $2", &ctx));
  EXPECT_THAT(result, StrEq("$0<no-subst-found> %1 $2<no-subst-found>"));
}

/// Allow flexible reference of positional parameters.
TEST(FormatTest, PositionalFmtStrFlexibleRef) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$2 $0 $2", &ctx, "a", "b", "c"));
  EXPECT_THAT(result, StrEq("c a c"));
}

TEST(FormatTest, PositionalFmtStrNoWhitespace) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("foo$0bar", &ctx, "-"));
  EXPECT_THAT(result, StrEq("foo-bar"));
}

TEST(FormatTest, PlaceHolderFmtStrWithSelf) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$_self", &ctx.withSelf("sss")));
  EXPECT_THAT(result, StrEq("sss"));
}

TEST(FormatTest, PlaceHolderFmtStrWithBuilder) {
  FmtContext ctx;

  std::string result = std::string(tgfmt("$_builder", &ctx.withBuilder("bbb")));
  EXPECT_THAT(result, StrEq("bbb"));
}

TEST(FormatTest, PlaceHolderFmtStrWithOp) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$_op", &ctx.withOp("ooo")));
  EXPECT_THAT(result, StrEq("ooo"));
}

TEST(FormatTest, PlaceHolderMissingCtx) {
  std::string result = std::string(tgfmt("$_op", nullptr));
  EXPECT_THAT(result, StrEq("$_op<no-subst-found>"));
}

TEST(FormatTest, PlaceHolderMissingSubst) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$_op", &ctx.withBuilder("builder")));
  EXPECT_THAT(result, StrEq("$_op<no-subst-found>"));
}

/// Test commonly used delimiters in C++.
TEST(FormatTest, PlaceHolderFmtStrDelimiter) {
  FmtContext ctx;
  ctx.addSubst("m", "");
  std::string result = std::string(tgfmt("$m{$m($m[$m]$m)$m}$m|", &ctx));
  EXPECT_THAT(result, StrEq("{([])}|"));
}

/// Test allowed characters in placeholder symbol.
TEST(FormatTest, CustomPlaceHolderFmtStrPlaceHolderChars) {
  FmtContext ctx;
  ctx.addSubst("m", "0 ");
  ctx.addSubst("m1", "1 ");
  ctx.addSubst("m2C", "2 ");
  ctx.addSubst("M_3", "3 ");
  std::string result = std::string(tgfmt("$m$m1$m2C$M_3", &ctx));
  EXPECT_THAT(result, StrEq("0 1 2 3 "));
}

TEST(FormatTest, CustomPlaceHolderFmtStrUnregisteredPlaceHolders) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("foo($awesome, $param)", &ctx));
  EXPECT_THAT(result,
              StrEq("foo($awesome<no-subst-found>, $param<no-subst-found>)"));
}

TEST(FormatTest, MixedFmtStr) {
  FmtContext ctx;
  ctx.withBuilder("bbb");

  std::string result = std::string(tgfmt("$_builder.build($_self, {$0, $1})",
                                         &ctx.withSelf("sss"), "a", "b"));
  EXPECT_THAT(result, StrEq("bbb.build(sss, {a, b})"));
}
