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

TEST(ArgParserTest, BasicCommand) {
  constexpr static Args::Command<> BasicCommand = {.name = "command"};
  auto args = Args::Parser<BasicCommand>::Parse({"a", "b", "c", "--", "--x--"},
                                                llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"), StrEq("--x--")));
}

constexpr Args::Flag TestFlag = {
    .name = "flag",
    .short_name = "f",
};
constexpr Args::StringOpt TestOpt = {
    .name = "option",
    .short_name = "o",
};

constexpr Args::Command<TestFlag, TestOpt> TestCommand = {
    .name = "command",
    .info = {},
};

TEST(ArgParserTest, GlobalCommand) {
  auto args = Args::Parser<TestCommand>::Parse(
      {"--flag", "a", "--option=test", "b", "c", "--", "--x--"}, llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag<TestFlag>());
  EXPECT_THAT(args.GetOption<TestOpt>(), Optional(StrEq("test")));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"), StrEq("--x--")));
}

TEST(ArgParserTest, EmptyPositional) {
  auto args = Args::Parser<TestCommand>::Parse({"a", "", "c", "--", "", "--x--"},
                                                llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq(""), StrEq("c"), StrEq(""),
                          StrEq("--x--")));
}

TEST(ArgParserTest, ShortNames) {
  constexpr static Args::Flag OtherFlag1 = {
      .name = "other-flag1",
      .short_name = "x",
  };
  constexpr static Args::Flag OtherFlag2 = {
      .name = "other-flag2",
      .short_name = "y",
  };
  constexpr static Args::Flag OtherFlag3 = {
      .name = "other-flag3",
      .short_name = "z",
  };
  constexpr static Args::Command<TestFlag, TestOpt, OtherFlag1, OtherFlag2,
                                 OtherFlag3>
      Command = {.name = "command"};
  auto args =
      Args::Parser<Command>::Parse({"a", "-xfyo=test", "b", "c"}, llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag<TestFlag>());
  EXPECT_TRUE(args.TestFlag<OtherFlag1>());
  EXPECT_TRUE(args.TestFlag<OtherFlag2>());
  EXPECT_FALSE(args.TestFlag<OtherFlag3>());
  EXPECT_THAT(args.GetOption<TestOpt>(), Optional(StrEq("test")));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

TEST(ArgParserTest, Overriding) {
  {
    auto args = Args::Parser<TestCommand>::Parse(
        {"--option=test1", "--option=test2"}, llvm::errs());
    EXPECT_TRUE(args);
    EXPECT_THAT(args.GetOption<TestOpt>(), Optional(StrEq("test2")));
  }
  {
    auto args = Args::Parser<TestCommand>::Parse(
        {"--option=test1", "-o=test2", "--option=test3"}, llvm::errs());
    EXPECT_TRUE(args);
    EXPECT_THAT(args.GetOption<TestOpt>(), Optional(StrEq("test3")));
  }
  {
    auto args = Args::Parser<TestCommand>::Parse(
        {"--option=test1", "--option=test2", "-o=test3"}, llvm::errs());
    EXPECT_TRUE(args);
    EXPECT_THAT(args.GetOption<TestOpt>(), Optional(StrEq("test3")));
  }
}

constexpr Args::Flag TestSubFlag = {.name = "flag"};
constexpr Args::StringOpt TestSubOpt = {.name = "option"};

enum class Subcommands {
  Sub1,
  Sub2,
  Sub3,
};

constexpr Args::Subcommand<Subcommands::Sub1, TestSubFlag> TestSub1 = {
    .name = "sub1",
};
constexpr Args::Subcommand<Subcommands::Sub2, TestSubFlag, TestSubOpt>
    TestSub2 = {
        .name = "sub2",
};

TEST(ArgParserTest, GlobalSubcommands) {
  auto args = Args::Parser<TestCommand, TestSub1, TestSub2>::Parse(
      {"--flag", "sub1", "a", "b", "c", "--", "--x--"}, llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag<TestFlag>());
  EXPECT_THAT(args.GetOption<TestOpt>(), Eq(std::nullopt));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"), StrEq("--x--")));
}

TEST(ArgParserTest, GlobalSubcommands2) {
  // Same opt spelling but in different parts of subcommand. Also repeated
  // opts and a value for the flag.
  auto args = Args::Parser<TestCommand, TestSub1, TestSub2>::Parse(
      {"--flag", "--option=main", "sub2", "a", "--flag", "--option=sub", "b",
       "c", "--flag=false"},
      llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_TRUE(args.TestFlag<TestFlag>());
  EXPECT_THAT(args.GetOption<TestOpt>(), Optional(StrEq("main")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub2));
  EXPECT_FALSE(args.TestFlag<TestSubFlag>());
  EXPECT_THAT(args.GetOption<TestSubOpt>(), Optional(StrEq("sub")));
  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

constexpr Args::Flag TestFlagDefault = {
    .name = "flag2",
    .default_value = true,
};
constexpr Args::StringOpt TestOptDefault = {
    .name = "option2",
    .default_value = "default",
};
constexpr Args::Flag TestSubFlagDefault = {
    .name = "flag2",
    .default_value = true,
};
constexpr Args::StringOpt TestSubOptDefault = {
    .name = "option2",
    .default_value = "default",
};
constexpr Args::Command<TestFlag, TestOpt, TestFlagDefault, TestOptDefault>
    TestCommand2 = {.name = "command"};
constexpr Args::Subcommand<Subcommands::Sub3, TestSubFlag, TestSubOpt,
                           TestSubFlagDefault, TestSubOptDefault>
    TestSub3 = {.name = "sub3"};

TEST(ArgParserTest, GlobalDefaults) {
  auto args = Args::Parser<TestCommand2, TestSub1, TestSub2, TestSub3>::Parse(
      {"sub3", "a", "b", "c"}, llvm::errs());
  EXPECT_TRUE(args);

  EXPECT_FALSE(args.TestFlag<TestFlag>());
  EXPECT_THAT(args.GetOption<TestOpt>(), Eq(std::nullopt));
  EXPECT_TRUE(args.TestFlag<TestFlagDefault>());
  EXPECT_THAT(args.GetOption<TestOptDefault>(), Optional(StrEq("default")));

  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub3));
  EXPECT_FALSE(args.TestFlag<TestSubFlag>());
  EXPECT_THAT(args.GetOption<TestSubOpt>(), Eq(std::nullopt));
  EXPECT_TRUE(args.TestFlag<TestSubFlagDefault>());
  EXPECT_THAT(args.GetOption<TestSubOptDefault>(), Optional(StrEq("default")));

  EXPECT_THAT(args.positional_args(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

TEST(ArgParserTest, DefaultsWithExplictOptions) {
  auto args = Args::Parser<TestCommand2, TestSub1, TestSub2, TestSub3>::Parse(
      {"--option2", "sub3", "a", "--option2=other", "b", "--option2", "c"},
      llvm::errs());
  EXPECT_TRUE(args);

  EXPECT_THAT(args.GetOption<TestOptDefault>(), Optional(StrEq("default")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub3));
  EXPECT_THAT(args.GetOption<TestSubOptDefault>(), Optional(StrEq("default")));
}

TEST(ArgParserTest, IntOptions) {
  constexpr static Args::IntOpt Opt = {.name = "int-opt"};
  constexpr static Args::IntOpt OptWithDefault = {
      .name = "int-defaulted-opt",
      .default_value = 42,
  };
  constexpr static Args::IntOpt SubOpt = {.name = "int-opt"};
  constexpr static Args::IntOpt SubOptWithDefault = {
      .name = "int-defaulted-opt",
      .default_value = 42,
  };

  constexpr static Args::Command<Opt, OptWithDefault> Command = {.name =
                                                                     "command"};
  constexpr static Args::Subcommand<Subcommands::Sub1, SubOpt,
                                    SubOptWithDefault>
      Subcommand = {.name = "sub"};

  auto args = Args::Parser<Command, Subcommand>::Parse(
      {"--int-opt=1", "sub", "--int-opt=2", "--int-defaulted-opt=3"},
      llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetOption<Opt>(), Optional(Eq(1)));
  EXPECT_THAT(args.GetOption<OptWithDefault>(), Optional(Eq(42)));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetOption<SubOpt>(), Optional(Eq(2)));
  EXPECT_THAT(args.GetOption<SubOptWithDefault>(), Optional(Eq(3)));
}

TEST(ArgParserTest, EnumOptions) {
  enum class OptEnum {
    Val1,
    Val2,
    Val3,
  };

  constexpr static std::tuple OptEnumValues = {
      Args::EnumOptValue<OptEnum::Val1>{.name = "val1"},
      Args::EnumOptValue<OptEnum::Val2>{.name = "val2"},
      Args::EnumOptValue<OptEnum::Val3>{.name = "val3"},
  };

  constexpr static Args::EnumOpt<OptEnumValues> Opt{
      .name = "enum-option1",
  };

  constexpr static Args::EnumOpt<OptEnumValues> SubOpt{
      .name = "enum-option1",
  };

  constexpr static Args::Command<Opt> Command = {
      .name = "command",
  };
  enum class Subcommands {
    Sub1,
  };
  constexpr static Args::Subcommand<Subcommands::Sub1, SubOpt> Subcommand = {
      .name = "sub",
  };
  auto args = Args::Parser<Command, Subcommand>::Parse(
      {"--enum-option1=val1", "sub", "--enum-option1=val2"}, llvm::errs());

  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetOption<Opt>(), Optional(Eq(OptEnum::Val1)));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetOption<SubOpt>(), Optional(Eq(OptEnum::Val2)));
}

TEST(ArgParserTest, StringListOption) {
  constexpr static Args::StringListOpt Opt = {.name = "strings1"};
  constexpr static Args::StringListOpt SubOpt = {.name = "strings1"};

  constexpr static Args::Command<Opt> Command = {.name = "command"};
  constexpr static Args::Subcommand<Subcommands::Sub1, SubOpt> Subcommand = {
      .name = "sub"};

  auto args = Args::Parser<Command, Subcommand>::Parse(
      {"--strings1=a", "--strings1=b", "sub", "--strings1=a", "--strings1=b",
       "--strings1=c"},
      llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetOption<Opt>(), ElementsAre(StrEq("a"), StrEq("b")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetOption<SubOpt>(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

TEST(ArgParserTest, StringListOptionDefaults) {
  constexpr static llvm::StringLiteral DefaultValues[] = {"a", "b", "c"};
  constexpr static Args::StringListOpt Opt = {
      .name = "strings1",
      .default_values = DefaultValues,
  };
  constexpr static Args::StringListOpt SubOpt = {
      .name = "strings1",
      .default_values = DefaultValues,
  };

  constexpr static Args::Command<Opt> Command = {.name = "command"};
  constexpr static Args::Subcommand<Subcommands::Sub1, SubOpt> Subcommand = {
      .name = "sub"};

  auto args = Args::Parser<Command, Subcommand>::Parse({"sub"}, llvm::errs());
  EXPECT_TRUE(args);
  EXPECT_THAT(args.GetOption<Opt>(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
  EXPECT_THAT(args.subcommand(), Eq(Subcommands::Sub1));
  EXPECT_THAT(args.GetOption<SubOpt>(),
              ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));
}

}  // namespace
}  // namespace Carbon::Testing
