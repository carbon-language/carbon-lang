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
using ::testing::Optional;
using ::testing::StrEq;

constexpr auto TestFlag = Args::MakeFlag("flag");
constexpr auto TestOpt = Args::MakeStringOpt("option");

constexpr auto TestCommand = Args::MakeCommand("command", &TestFlag, &TestOpt);

TEST(ArgParserTest, GlobalCommand) {
  auto args =
      Args::Parse({"--flag", "a", "--option=test", "b", "c", "--", "--x--"},
                  llvm::errs(), TestCommand);

  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag(&TestFlag));
  EXPECT_THAT(args.GetStringOpt(&TestOpt), Optional(StrEq("test")));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"), StrEq("--x--")));
}

constexpr auto TestSubFlag = Args::MakeFlag("flag");
constexpr auto TestSubOpt = Args::MakeStringOpt("option");

enum class Subcommands {
  Sub1,
  Sub2,
  Sub3,
};

constexpr auto TestSub1 =
    Args::MakeSubcommand("sub1", Subcommands::Sub1, &TestSubFlag);
constexpr auto TestSub2 =
    Args::MakeSubcommand("sub2", Subcommands::Sub2, &TestSubFlag, &TestSubOpt);

TEST(ArgParserTest, GlobalSubcommands) {
  auto args = Args::Parse({"--flag", "sub1", "a", "b", "c", "--", "--x--"},
                          llvm::errs(), TestCommand, TestSub1, TestSub2);
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag(&TestFlag));
  EXPECT_THAT(args.GetStringOpt(&TestOpt), Eq(std::nullopt));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"), StrEq("--x--")));
}

TEST(ArgParserTest, GlobalSubcommands2) {
  // Same opt spelling but in different parts of subcommand. Also repeated
  // opts and a value for the flag.
  auto args = Args::Parse({"--flag", "--option=main", "sub2", "a", "--flag",
                           "--option=sub", "b", "c", "--flag=false"},
                          llvm::errs(), TestCommand, TestSub1, TestSub2);
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag(&TestFlag));
  EXPECT_THAT(args.GetStringOpt(&TestOpt), Optional(StrEq("main")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub2));
  EXPECT_FALSE(args.TestSubcommandFlag(&TestSubFlag));
  EXPECT_THAT(args.GetSubcommandStringOpt(&TestSubOpt), Optional(StrEq("sub")));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

constexpr auto TestFlagDefault =
    Args::MakeFlag("flag2", /*short_name=*/"", /*default_value=*/true);
constexpr auto TestOptDefault = Args::MakeStringOpt(
    "option2", /*short_name=*/"", /*default_value=*/"default");
constexpr auto TestCommand2 = Args::MakeCommand(
    "command", &TestFlag, &TestOpt, &TestFlagDefault, &TestOptDefault);
constexpr auto TestSub3 =
    Args::MakeSubcommand("sub3", Subcommands::Sub3, &TestSubFlag, &TestSubOpt,
                         &TestFlagDefault, &TestOptDefault);

TEST(ArgParserTest, GlobalDefaults) {
  auto args = Args::Parse({"sub3", "a", "b", "c"}, llvm::errs(), TestCommand2,
                          TestSub1, TestSub2, TestSub3);
  EXPECT_TRUE(args);

  EXPECT_FALSE(args.TestFlag(&TestFlag));
  EXPECT_THAT(args.GetStringOpt(&TestOpt), Eq(std::nullopt));
  EXPECT_TRUE(args.TestFlag(&TestFlagDefault));
  EXPECT_THAT(args.GetStringOpt(&TestOptDefault), Optional(StrEq("default")));

  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub3));
  EXPECT_FALSE(args.TestSubcommandFlag(&TestSubFlag));
  EXPECT_THAT(args.GetSubcommandStringOpt(&TestSubOpt), Eq(std::nullopt));
  EXPECT_TRUE(args.TestSubcommandFlag(&TestFlagDefault));
  EXPECT_THAT(args.GetSubcommandStringOpt(&TestOptDefault),
              Optional(StrEq("default")));

  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

TEST(ArgParserTest, DefaultsWithExplictOptions) {
  auto args = Args::Parse(
      {"--option2", "sub3", "a", "--option2=other", "b", "--option2", "c"},
      llvm::errs(), TestCommand2, TestSub1, TestSub2, TestSub3);
  EXPECT_TRUE(args);

  EXPECT_THAT(args.GetStringOpt(&TestOptDefault), Optional(StrEq("default")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub3));
  EXPECT_THAT(args.GetSubcommandStringOpt(&TestOptDefault),
              Optional(StrEq("default")));
}

TEST(ArgParserTest, IntOptions) {
  constexpr static auto Opt = Args::MakeIntOpt("int-opt");
  constexpr static auto OptWithDefault =
      Args::MakeIntOpt("int-defaulted-opt", /*short_name=*/"", /*default_value=*/42);

  constexpr auto Command = Args::MakeCommand("command", &Opt, &OptWithDefault);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Opt, &OptWithDefault);

  auto args = Args::Parse(
      {"--int-opt=1", "sub", "--int-opt=2", "--int-defaulted-opt=3"},
      llvm::errs(), Command, Subcommand);
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetIntOpt(&Opt), Optional(Eq(1)));
  EXPECT_THAT(args.GetIntOpt(&OptWithDefault), Optional(Eq(42)));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandIntOpt(&Opt), Optional(Eq(2)));
  EXPECT_THAT(args.GetSubcommandIntOpt(&OptWithDefault), Optional(Eq(3)));
}

TEST(ArgParserTest, EnumOptions) {
  enum class OptEnum {
    Val1,
    Val2,
    Val3,
  };

  constexpr static auto Opt = Args::MakeEnumOpt<OptEnum>(
      "enum-flag1", {
                        {.name = "val1", .value = OptEnum::Val1},
                        {.name = "val2", .value = OptEnum::Val2},
                        {.name = "val3", .value = OptEnum::Val3},
                    });

  constexpr auto Command = Args::MakeCommand("command", &Opt);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Opt);
  auto args = Args::Parse({"--enum-flag1=val1", "sub", "--enum-flag1=val2"},
                          llvm::errs(), Command, Subcommand);

  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetEnumOpt(&Opt), Optional(Eq(OptEnum::Val1)));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandEnumOpt(&Opt), Optional(Eq(OptEnum::Val2)));
}

TEST(ArgParserTest, StringListOption) {
  constexpr static auto Opt = Args::MakeStringListOpt("strings1");

  constexpr auto Command = Args::MakeCommand("command", &Opt);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Opt);

  auto args = Args::Parse({"--strings1=a", "--strings1=b", "sub",
                           "--strings1=a", "--strings1=b", "--strings1=c"},
                          llvm::errs(), Command, Subcommand);
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetStringListOpt(&Opt), ElementsAre(StrEq("a"), StrEq("b")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandStringListOpt(&Opt),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

TEST(ArgParserTest, StringListOptionDefaults) {
  constexpr static llvm::StringRef DefaultValues[] = {"a", "b", "c"};
  constexpr static auto Opt =
      Args::MakeStringListOpt("strings1", /*short_name=*/"", /*default_values=*/DefaultValues);

  constexpr auto Command = Args::MakeCommand("command", &Opt);
  constexpr auto Subcommand =
      Args::MakeSubcommand("sub", Subcommands::Sub1, &Opt);

  auto args = Args::Parse({"sub"}, llvm::errs(), Command, Subcommand);
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetStringListOpt(&Opt),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetSubcommandStringListOpt(&Opt),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

}  // namespace
}  // namespace Carbon::Testing
