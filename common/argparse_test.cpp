// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/argparse.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/test_raw_ostream.h"
#include "gmock/gmock.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::StrEq;
using ::testing::Optional;

constexpr auto TestFlag1 = Args::MakeBooleanFlag("flag1");
constexpr auto TestFlag2 = Args::MakeStringFlag("flag2");

constexpr auto TestCommand =
    Args::MakeCommand("command", &TestFlag1, &TestFlag2);

TEST(ArgParserTest, GlobalCommand) {
  auto args = Args::Parse(
      {"--flag1", "a", "--flag2=test", "b", "c", "--", "--x--"}, llvm::errs(),
      TestCommand);

  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag(&TestFlag1));
  EXPECT_THAT(args.GetStringFlag(&TestFlag2), Optional(StrEq("test")));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"), StrEq("--x--")));
}

constexpr auto TestSubFlag1 = Args::MakeBooleanFlag("flag1");
constexpr auto TestSubFlag2 = Args::MakeStringFlag("flag2");

enum class Subcommands {
  Sub1,
  Sub2,
  Sub3,
};

constexpr auto TestSub1 =
    Args::MakeSubcommand("sub1", Subcommands::Sub1, &TestSubFlag1);
constexpr auto TestSub2 = Args::MakeSubcommand(
    "sub2", Subcommands::Sub2, &TestSubFlag1, &TestSubFlag2);

TEST(ArgParserTest, GlobalSubcommands) {
  auto args = Args::Parse(
      {"--flag1", "sub1", "a", "b", "c", "--", "--x--"},
      llvm::errs(), TestCommand, TestSub1, TestSub2);
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag(&TestFlag1));
  EXPECT_THAT(args.GetStringFlag(&TestFlag2), Eq(std::nullopt));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"), StrEq("--x--")));
}

TEST(ArgParserTest, GlobalSubcommands2) {
  // Same flag spelling but in different parts of subcommand. Also repeated
  // flags and a value for a boolean.
  auto args = Args::Parse(
      {"--flag1", "--flag2=main", "sub2", "a", "--flag1", "--flag2=sub", "b", "c", "--flag1=false"},
      llvm::errs(), TestCommand, TestSub1, TestSub2);
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag(&TestFlag1));
  EXPECT_THAT(args.GetStringFlag(&TestFlag2), Optional(StrEq("main")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub2));
  EXPECT_FALSE(args.TestSubcommandFlag(&TestSubFlag1));
  EXPECT_THAT(args.GetSubcommandStringFlag(&TestSubFlag2), Optional(StrEq("sub")));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

constexpr auto TestFlag3 = Args::MakeBooleanFlag("flag3", {.default_value = true});
constexpr auto TestFlag4 = Args::MakeStringFlag("flag4", {.default_value = "default"});
constexpr auto TestCommand2 =
    Args::MakeCommand("command", &TestFlag1, &TestFlag2, &TestFlag3, &TestFlag4);
constexpr auto TestSub3 = Args::MakeSubcommand(
    "sub3", Subcommands::Sub3, &TestSubFlag1, &TestSubFlag2, &TestFlag3, &TestFlag4);

TEST(ArgParserTest, GlobalDefaults) {
  auto args = Args::Parse({"sub3", "a", "b", "c"},
      llvm::errs(), TestCommand2, TestSub1, TestSub2, TestSub3);
  EXPECT_TRUE(args);

  EXPECT_FALSE(args.TestFlag(&TestFlag1));
  EXPECT_THAT(args.GetStringFlag(&TestFlag2), Eq(std::nullopt));
  EXPECT_TRUE(args.TestFlag(&TestFlag3));
  EXPECT_THAT(args.GetStringFlag(&TestFlag4), Optional(StrEq("default")));

  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub3));
  EXPECT_FALSE(args.TestSubcommandFlag(&TestSubFlag1));
  EXPECT_THAT(args.GetSubcommandStringFlag(&TestSubFlag2), Eq(std::nullopt));
  EXPECT_TRUE(args.TestSubcommandFlag(&TestFlag3));
  EXPECT_THAT(args.GetSubcommandStringFlag(&TestFlag4), Optional(StrEq("default")));

  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

TEST(ArgParserTest, DefaultsWithExplictFlags) {
  auto args = Args::Parse({"--flag4", "sub3", "a", "--flag4=other", "b", "--flag4", "c"},
      llvm::errs(), TestCommand2, TestSub1, TestSub2, TestSub3);
  EXPECT_TRUE(args);

  EXPECT_THAT(args.GetStringFlag(&TestFlag4), Optional(StrEq("default")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub3));
  EXPECT_THAT(args.GetSubcommandStringFlag(&TestFlag4), Optional(StrEq("default")));
}

TEST(ArgParserTest, IntFlags) {
  constexpr static auto Flag = Args::MakeIntFlag("int-flag");
  constexpr static auto FlagWithDefault =
      Args::MakeIntFlag("int-defaulted-flag", {.default_value = 42});

  constexpr auto Command =
      Args::MakeCommand("command", &Flag, &FlagWithDefault);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Flag, &FlagWithDefault);

  auto args = Args::Parse(
      {"--int-flag=1", "sub", "--int-flag=2", "--int-defaulted-flag=3"},
      llvm::errs(), Command, Subcommand);
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetIntFlag(&Flag), Optional(Eq(1)));
  EXPECT_THAT(args.GetIntFlag(&FlagWithDefault), Optional(Eq(42)));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandIntFlag(&Flag), Optional(Eq(2)));
  EXPECT_THAT(args.GetSubcommandIntFlag(&FlagWithDefault), Optional(Eq(3)));
}

TEST(ArgParserTest, EnumFlags) {
  enum class FlagEnum {
    Val1,
    Val2,
    Val3,
  };

  constexpr static auto Flag = Args::MakeEnumFlag<FlagEnum>(
      "enum-flag1", {
                        {.name = "val1", .value = FlagEnum::Val1},
                        {.name = "val2", .value = FlagEnum::Val2},
                        {.name = "val3", .value = FlagEnum::Val3},
                    });

  constexpr auto Command = Args::MakeCommand("command", &Flag);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Flag);
  auto args = Args::Parse({"--enum-flag1=val1", "sub", "--enum-flag1=val2"},
                          llvm::errs(), Command, Subcommand);

  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetEnumFlag(&Flag), Optional(Eq(FlagEnum::Val1)));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandEnumFlag(&Flag), Optional(Eq(FlagEnum::Val2)));
}

TEST(ArgParserTest, StringListFlag) {
  constexpr static auto Flag = Args::MakeStringListFlag("strings1");

  constexpr auto Command = Args::MakeCommand("command", &Flag);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Flag);

  auto args = Args::Parse({"--strings1=a", "--strings1=b", "sub",
                           "--strings1=a", "--strings1=b", "--strings1=c"},
                          llvm::errs(), Command, Subcommand);
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetStringListFlag(&Flag),
              ElementsAre(StrEq("a"), StrEq("b")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandStringListFlag(&Flag),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

TEST(ArgParserTest, StringListFlagDefaults) {
  constexpr static llvm::StringLiteral DefaultValues[] = {"a", "b", "c"};
  constexpr static auto Flag =
      Args::MakeStringListFlag("strings1", {.default_values = DefaultValues});

  constexpr auto Command = Args::MakeCommand("command", &Flag);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Flag);

  auto args = Args::Parse({"sub"}, llvm::errs(), Command, Subcommand);
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetStringListFlag(&Flag),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandStringListFlag(&Flag),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

}  // namespace
}  // namespace Carbon::Testing
