// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/command_line.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/error_test_helpers.h"
#include "common/raw_string_ostream.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::CommandLine {
namespace {

using ::Carbon::Testing::IsError;
using ::Carbon::Testing::IsSuccess;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::StrEq;

constexpr CommandInfo TestCommandInfo = {
    .name = "test",
    .help = "A test command.",
    .help_epilogue = "TODO",
};

enum class TestEnum : int8_t {
  Val1,
  Val2,
};

enum class TestSubcommand : int8_t {
  Sub1,
  Sub2,
};

TEST(ArgParserTest, BasicCommand) {
  bool flag = false;
  int integer_option = -1;
  llvm::StringRef string_option = "";
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args) {
    return Parse(args, llvm::errs(), TestCommandInfo, [&](auto& b) {
      b.AddFlag({.name = "flag"}, [&](auto& arg_b) { arg_b.Set(&flag); });
      b.AddIntegerOption({.name = "option1"},
                         [&](auto& arg_b) { arg_b.Set(&integer_option); });
      b.AddStringOption({.name = "option2"},
                        [&](auto& arg_b) { arg_b.Set(&string_option); });
      b.Do([] {});
    });
  };

  EXPECT_THAT(parse({"--flag", "--option2=value", "--option1=42"}),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_THAT(integer_option, Eq(42));
  EXPECT_THAT(string_option, StrEq("value"));
}

TEST(ArgParserTest, BooleanFlags) {
  bool flag = false;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddFlag({.name = "flag"}, [&](auto& arg_b) { arg_b.Set(&flag); });
      b.Do([] {});
    });
  };

  EXPECT_THAT(parse({"--no-flag"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_FALSE(flag);

  EXPECT_THAT(parse({"--flag", "--no-flag"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_FALSE(flag);

  EXPECT_THAT(parse({"--no-flag", "--flag"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);

  EXPECT_THAT(parse({"--no-flag", "--flag", "--flag=false"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_FALSE(flag);

  EXPECT_THAT(parse({"--no-flag", "--flag=true"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);

  EXPECT_THAT(parse({"--no-flag", "--flag=true", "--no-flag"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_FALSE(flag);

  RawStringOstream os;
  EXPECT_THAT(
      parse({"--no-flag=true"}, os),
      IsError(StrEq("cannot specify a value when using a flag name prefixed "
                    "with `no-` -- that prefix implies a value of `false`")));

  EXPECT_THAT(
      parse({"--no-flag=false"}, os),
      IsError(StrEq("cannot specify a value when using a flag name prefixed "
                    "with `no-` -- that prefix implies a value of `false`")));
}

TEST(ArgParserTest, ArgDefaults) {
  bool flag = false;
  int integer_option = -1;
  llvm::StringRef string_option = "";
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddFlag({.name = "flag"}, [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&flag);
      });
      b.AddIntegerOption({.name = "option1"}, [&](auto& arg_b) {
        arg_b.Default(7);
        arg_b.Set(&integer_option);
      });
      b.AddStringOption({.name = "option2"}, [&](auto& arg_b) {
        arg_b.Default("default");
        arg_b.Set(&string_option);
      });
      b.Do([] {});
    });
  };

  EXPECT_THAT(parse({}, llvm::errs()), IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_THAT(integer_option, Eq(7));
  EXPECT_THAT(string_option, StrEq("default"));

  EXPECT_THAT(parse({"--option1", "--option2"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_THAT(integer_option, Eq(7));
  EXPECT_THAT(string_option, StrEq("default"));

  EXPECT_THAT(parse({"--option1=42", "--option2=explicit"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_THAT(integer_option, Eq(42));
  EXPECT_THAT(string_option, StrEq("explicit"));

  EXPECT_THAT(
      parse({"--option1=42", "--option2=explicit", "--option1", "--option2"},
            llvm::errs()),
      IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_THAT(integer_option, Eq(7));
  EXPECT_THAT(string_option, StrEq("default"));
}

TEST(ArgParserTest, ShortArgs) {
  bool flag = false;
  bool example = false;
  int integer_option = -1;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddFlag({.name = "flag", .short_name = "f"},
                [&](auto& arg_b) { arg_b.Set(&flag); });
      b.AddFlag({.name = "example", .short_name = "x"},
                [&](auto& arg_b) { arg_b.Set(&example); });
      b.AddIntegerOption({.name = "option1", .short_name = "o"},
                         [&](auto& arg_b) {
                           arg_b.Default(123);
                           arg_b.Set(&integer_option);
                         });
      b.AddIntegerOption({.name = "option2", .short_name = "z"},
                         [&](auto& arg_b) { arg_b.Set(&integer_option); });
      b.Do([] {});
    });
  };

  EXPECT_THAT(parse({"-f", "-o=42", "-x"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_TRUE(example);
  EXPECT_THAT(integer_option, Eq(42));

  flag = false;
  example = false;
  EXPECT_THAT(parse({"--option1=13", "-xfo"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_TRUE(example);
  EXPECT_THAT(integer_option, Eq(123));

  RawStringOstream os;

  EXPECT_THAT(parse({"-v"}, os), IsError(StrEq("unknown short option `-v`")));
  EXPECT_THAT(parse({"-xvx"}, os), IsError(StrEq("unknown short option `-v`")));

  EXPECT_THAT(
      parse({"-xzf"}, os),
      IsError(StrEq("option `-z` (short for `--option2`) requires a value to "
                    "be provided and none was")));

  EXPECT_THAT(
      parse({"--option2"}, os),
      IsError(StrEq(
          "option `--option2` requires a value to be provided and none was")));

  EXPECT_THAT(
      parse({"-xz=123"}, os),
      IsError(StrEq("cannot provide a value to the group of multiple short "
                    "options `-xz=...`; values must be provided to a single "
                    "option, using either the short or long spelling")));
}

TEST(ArgParserTest, PositionalArgs) {
  bool flag = false;
  llvm::StringRef string_option = "";
  llvm::StringRef source_string;
  llvm::StringRef dest_string;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddFlag({.name = "flag"}, [&](auto& arg_b) { arg_b.Set(&flag); });
      b.AddStringOption({.name = "option"},
                        [&](auto& arg_b) { arg_b.Set(&string_option); });
      b.AddStringPositionalArg({.name = "source"}, [&](auto& arg_b) {
        arg_b.Set(&source_string);
        arg_b.Required(true);
      });
      b.AddStringPositionalArg({.name = "dest"}, [&](auto& arg_b) {
        arg_b.Set(&dest_string);
        arg_b.Required(true);
      });
      b.Do([] {});
    });
  };

  RawStringOstream os;
  EXPECT_THAT(parse({"--flag", "--option=x"}, os),
              IsError(StrEq(
                  "not all required positional arguments were provided; first "
                  "missing and required positional argument: `source`")));

  EXPECT_THAT(parse({"src", "--flag", "--option=value", "dst"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_THAT(string_option, StrEq("value"));
  EXPECT_THAT(source_string, StrEq("src"));
  EXPECT_THAT(dest_string, StrEq("dst"));

  EXPECT_THAT(parse({"src2", "--", "dst2"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(source_string, StrEq("src2"));
  EXPECT_THAT(dest_string, StrEq("dst2"));

  EXPECT_THAT(parse({"-", "--", "-"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(source_string, StrEq("-"));
  EXPECT_THAT(dest_string, StrEq("-"));
}

TEST(ArgParserTest, PositionalAppendArgs) {
  bool flag = false;
  llvm::StringRef string_option = "";
  llvm::SmallVector<llvm::StringRef> source_strings;
  llvm::SmallVector<llvm::StringRef> dest_strings;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddFlag({.name = "flag"}, [&](auto& arg_b) { arg_b.Set(&flag); });
      b.AddStringOption({.name = "option"},
                        [&](auto& arg_b) { arg_b.Set(&string_option); });
      b.AddStringPositionalArg({.name = "sources"}, [&](auto& arg_b) {
        arg_b.Append(&source_strings);
        arg_b.Required(true);
      });
      b.AddStringPositionalArg({.name = "dest"}, [&](auto& arg_b) {
        arg_b.Append(&dest_strings);
        arg_b.Required(true);
      });
      b.Do([] {});
    });
  };

  RawStringOstream os;
  EXPECT_THAT(parse({"--flag", "--option=x"}, os),
              IsError(StrEq(
                  "not all required positional arguments were provided; first "
                  "missing and required positional argument: `sources`")));

  EXPECT_THAT(
      parse({"src1", "--flag", "src2", "--option=value", "--", "--dst--"},
            llvm::errs()),
      IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(flag);
  EXPECT_THAT(string_option, StrEq("value"));
  EXPECT_THAT(source_strings, ElementsAre(StrEq("src1"), StrEq("src2")));
  EXPECT_THAT(dest_strings, ElementsAre(StrEq("--dst--")));

  source_strings.clear();
  dest_strings.clear();
  EXPECT_THAT(
      parse({"--", "--src1--", "--src2--", "--", "dst1", "dst2"}, llvm::errs()),
      IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(source_strings,
              ElementsAre(StrEq("--src1--"), StrEq("--src2--")));
  EXPECT_THAT(dest_strings, ElementsAre(StrEq("dst1"), StrEq("dst2")));

  source_strings.clear();
  dest_strings.clear();
  EXPECT_THAT(parse({"--", "--", "dst1", "dst2"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(source_strings, ElementsAre());
  EXPECT_THAT(dest_strings, ElementsAre(StrEq("dst1"), StrEq("dst2")));
}

TEST(ArgParserTest, BasicSubcommands) {
  bool flag = false;
  llvm::StringRef option1 = "";
  llvm::StringRef option2 = "";

  TestSubcommand subcommand;
  bool subsub_command = false;

  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddSubcommand({.name = "sub1"}, [&](auto& sub_b) {
        sub_b.AddFlag({.name = "flag"}, [&](auto& arg_b) { arg_b.Set(&flag); });
        sub_b.AddStringOption({.name = "option"},
                              [&](auto& arg_b) { arg_b.Set(&option1); });
        sub_b.Do([&] { subcommand = TestSubcommand::Sub1; });
      });
      b.AddSubcommand({.name = "sub2"}, [&](auto& sub_b) {
        sub_b.AddStringOption({.name = "option"},
                              [&](auto& arg_b) { arg_b.Set(&option1); });
        sub_b.AddSubcommand({.name = "subsub"}, [&](auto& subsub_b) {
          subsub_b.AddStringOption({.name = "option"},
                                   [&](auto& arg_b) { arg_b.Set(&option2); });
          subsub_b.Do([&] { subsub_command = true; });
        });
        sub_b.Do([&] { subcommand = TestSubcommand::Sub2; });
      });
      b.RequiresSubcommand();
    });
  };

  RawStringOstream os;
  EXPECT_THAT(parse({}, os),
              IsError(StrEq("no subcommand specified; available subcommands: "
                            "`sub1`, `sub2`, or `help`")));

  EXPECT_THAT(parse({"--flag"}, os), IsError(StrEq("unknown option `--flag`")));

  EXPECT_THAT(parse({"sub3"}, os),
              IsError(StrEq("invalid subcommand `sub3`; available subcommands: "
                            "`sub1`, `sub2`, or `help`")));

  EXPECT_THAT(parse({"--flag", "sub1"}, os),
              IsError(StrEq("unknown option `--flag`")));

  EXPECT_THAT(parse({"sub1", "--flag"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(subcommand, Eq(TestSubcommand::Sub1));
  EXPECT_TRUE(flag);

  EXPECT_THAT(parse({"sub2", "--option=value"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(subcommand, Eq(TestSubcommand::Sub2));
  EXPECT_THAT(option1, StrEq("value"));

  EXPECT_THAT(parse({"sub2", "--option=abc", "subsub42", "--option=xyz"}, os),
              IsError(StrEq("invalid subcommand `subsub42`; available "
                            "subcommands: `subsub` or `help`")));

  EXPECT_THAT(
      parse({"sub2", "--option=abc", "subsub", "--option=xyz"}, llvm::errs()),
      IsSuccess(Eq(ParseResult::Success)));
  EXPECT_TRUE(subsub_command);
  EXPECT_THAT(option1, StrEq("abc"));
  EXPECT_THAT(option2, StrEq("xyz"));
}

TEST(ArgParserTest, Appending) {
  llvm::SmallVector<int> integers;
  llvm::SmallVector<llvm::StringRef> strings;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddIntegerOption({.name = "int"},
                         [&](auto& arg_b) { arg_b.Append(&integers); });
      b.AddStringOption({.name = "str"},
                        [&](auto& arg_b) { arg_b.Append(&strings); });
      b.Do([] {});
    });
  };

  EXPECT_THAT(parse({}, llvm::errs()), IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(integers, ElementsAre());
  EXPECT_THAT(strings, ElementsAre());

  EXPECT_THAT(parse({"--int=1", "--int=2", "--int=3"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(integers, ElementsAre(Eq(1), Eq(2), Eq(3)));
  EXPECT_THAT(strings, ElementsAre());

  EXPECT_THAT(parse({"--str=a", "--str=b", "--str=c"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(integers, ElementsAre(Eq(1), Eq(2), Eq(3)));
  EXPECT_THAT(strings, ElementsAre(StrEq("a"), StrEq("b"), StrEq("c")));

  EXPECT_THAT(parse({"--str=d", "--int=4", "--str=e"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(integers, ElementsAre(Eq(1), Eq(2), Eq(3), Eq(4)));
  EXPECT_THAT(strings, ElementsAre(StrEq("a"), StrEq("b"), StrEq("c"),
                                   StrEq("d"), StrEq("e")));
}

TEST(ArgParserTest, PositionalAppending) {
  llvm::SmallVector<llvm::StringRef> option_strings;
  llvm::SmallVector<llvm::StringRef> strings1;
  llvm::SmallVector<llvm::StringRef> strings2;
  llvm::SmallVector<llvm::StringRef> strings3;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddStringOption({.name = "opt"},
                        [&](auto& arg_b) { arg_b.Append(&option_strings); });
      b.AddStringPositionalArg({.name = "s1"},
                               [&](auto& arg_b) { arg_b.Append(&strings1); });
      b.AddStringPositionalArg({.name = "s2"},
                               [&](auto& arg_b) { arg_b.Append(&strings2); });
      b.AddStringPositionalArg({.name = "s3"},
                               [&](auto& arg_b) { arg_b.Append(&strings3); });
      b.Do([] {});
    });
  };

  EXPECT_THAT(parse({"a", "--opt=x", "b", "--opt=y", "--", "c", "--opt=z", "d",
                     "--", "e", "f"},
                    llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(option_strings, ElementsAre(StrEq("x"), StrEq("y")));
  EXPECT_THAT(strings1, ElementsAre(StrEq("a"), StrEq("b")));
  EXPECT_THAT(strings2, ElementsAre(StrEq("c"), StrEq("--opt=z"), StrEq("d")));
  EXPECT_THAT(strings3, ElementsAre(StrEq("e"), StrEq("f")));
}

static auto ParseOneOfOption(
    llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s,
    llvm::function_ref<auto(OneOfArgBuilder&)->void> build)
    -> ErrorOr<ParseResult> {
  return Parse(args, s, TestCommandInfo, [&](auto& b) {
    b.AddOneOfOption({.name = "option"}, build);
    b.Do([] {});
  });
}

TEST(ArgParserTest, OneOfOption) {
  int value = 0;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return ParseOneOfOption(args, s, [&](auto& arg_b) {
      arg_b.SetOneOf(
          {
              arg_b.OneOfValue("x", 1),
              arg_b.OneOfValue("y", 2),
              arg_b.OneOfValue("z", 3),
          },
          &value);
    });
  };

  EXPECT_THAT(parse({"--option=x"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(value, Eq(1));

  EXPECT_THAT(parse({"--option=y"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(value, Eq(2));

  EXPECT_THAT(parse({"--option=z"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(value, Eq(3));

  RawStringOstream os;

  EXPECT_THAT(
      parse({"--option"}, os),
      IsError(StrEq(
          "option `--option` requires a value to be provided and none was")));

  constexpr const char* ErrorStr =
      "option `--option={0}` has an invalid value `{0}`; valid values are: "
      "`x`, `y`, or `z`";
  EXPECT_THAT(parse({"--option=a"}, os),
              IsError(StrEq(llvm::formatv(ErrorStr, "a"))));

  EXPECT_THAT(parse({"--option="}, os),
              IsError(StrEq(llvm::formatv(ErrorStr, ""))));

  EXPECT_THAT(parse({"--option=xx"}, os),
              IsError(StrEq(llvm::formatv(ErrorStr, "xx"))));

  EXPECT_THAT(parse({"--option=\xFF"}, os),
              IsError(StrEq(llvm::formatv(ErrorStr, "\\FF"))));

  EXPECT_THAT(parse({llvm::StringRef("--option=\0", 10)}, os),
              IsError(StrEq(llvm::formatv(ErrorStr, "\\00"))));
}

TEST(ArgParserTest, OneOfOptionWithTwoOptions) {
  int value = 0;
  RawStringOstream os;
  EXPECT_THAT(ParseOneOfOption({"--option=z"}, os,
                               [&](auto& arg_b) {
                                 arg_b.SetOneOf(
                                     {
                                         arg_b.OneOfValue("x", 1),
                                         arg_b.OneOfValue("y", 2),
                                     },
                                     &value);
                               }),
              IsError(StrEq("option `--option=z` has an invalid value `z`; "
                            "valid values are: `x` or `y`")));
}

TEST(ArgParserTest, OneOfOptionWithOneOption) {
  int value = 0;
  RawStringOstream os;
  EXPECT_THAT(ParseOneOfOption({"--option=z"}, os,
                               [&](auto& arg_b) {
                                 arg_b.SetOneOf(
                                     {
                                         arg_b.OneOfValue("x", 1),
                                     },
                                     &value);
                               }),
              IsError(StrEq("option `--option=z` has an invalid value `z`; "
                            "valid values are: `x`")));
}

TEST(ArgParserTest, OneOfOptionAppending) {
  llvm::SmallVector<int> values;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return ParseOneOfOption(args, s, [&](auto& arg_b) {
      arg_b.AppendOneOf(
          {
              arg_b.OneOfValue("x", 1),
              arg_b.OneOfValue("y", 2),
              arg_b.OneOfValue("z", 3),
          },
          &values);
    });
  };

  EXPECT_THAT(parse({}, llvm::errs()), IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(values, ElementsAre());

  EXPECT_THAT(parse({"--option=x", "--option=y", "--option=z"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(values, ElementsAre(Eq(1), Eq(2), Eq(3)));

  EXPECT_THAT(parse({"--option=y", "--option=y", "--option=x"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(values, ElementsAre(Eq(1), Eq(2), Eq(3), Eq(2), Eq(2), Eq(1)));
}

TEST(ArgParserTest, RequiredArgs) {
  int integer_option;
  llvm::StringRef string_option;

  TestSubcommand subcommand;
  int integer_sub_option;
  llvm::StringRef string_sub_option;
  TestEnum enum_sub_option;

  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(args, s, TestCommandInfo, [&](auto& b) {
      b.AddIntegerOption({.name = "opt1"}, [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Set(&integer_option);
      });
      b.AddStringOption({.name = "opt2"}, [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Set(&string_option);
      });
      b.AddSubcommand({.name = "sub1"}, [&](auto& sub_b) {
        sub_b.AddIntegerOption({.name = "sub-opt1"}, [&](auto& arg_b) {
          arg_b.Required(true);
          arg_b.Set(&integer_sub_option);
        });
        sub_b.AddStringOption({.name = "sub-opt2"}, [&](auto& arg_b) {
          arg_b.Required(true);
          arg_b.Set(&string_sub_option);
        });
        sub_b.Do([&] { subcommand = TestSubcommand::Sub1; });
      });
      b.AddSubcommand({.name = "sub2"}, [&](auto& sub_b) {
        sub_b.AddIntegerOption({.name = "sub-opt1"}, [&](auto& arg_b) {
          arg_b.Required(true);
          arg_b.Set(&integer_sub_option);
        });
        sub_b.AddStringOption({.name = "sub-opt2"}, [&](auto& arg_b) {
          arg_b.Required(true);
          arg_b.Set(&string_sub_option);
        });
        sub_b.AddOneOfOption({.name = "sub-opt3"}, [&](auto& arg_b) {
          arg_b.Required(true);
          arg_b.SetOneOf(
              {
                  arg_b.OneOfValue("a", TestEnum::Val1),
                  arg_b.OneOfValue("b", TestEnum::Val2),
              },
              &enum_sub_option);
        });
        sub_b.Do([&] { subcommand = TestSubcommand::Sub2; });
      });
      b.Do([] {});
    });
  };

  RawStringOstream os;
  EXPECT_THAT(parse({}, os),
              IsError(StrEq("required options not provided: --opt1, --opt2")));

  EXPECT_THAT(parse({"--opt2=xyz"}, os),
              IsError(StrEq("required options not provided: --opt1")));

  EXPECT_THAT(parse({"--opt2=xyz", "--opt1=42"}, llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(integer_option, Eq(42));
  EXPECT_THAT(string_option, StrEq("xyz"));

  EXPECT_THAT(parse({"--opt2=xyz", "--opt1=42", "sub2"}, os),
              IsError(StrEq("required options not provided: --sub-opt1, "
                            "--sub-opt2, --sub-opt3")));

  EXPECT_THAT(
      parse({"--opt2=xyz", "--opt1=42", "sub2", "--sub-opt3=b"}, os),
      IsError(StrEq("required options not provided: --sub-opt1, --sub-opt2")));

  EXPECT_THAT(parse({"--opt2=xyz", "--opt1=42", "sub2", "--sub-opt3=b",
                     "--sub-opt1=13"},
                    os),
              IsError(StrEq("required options not provided: --sub-opt2")));

  EXPECT_THAT(parse({"--opt2=xyz", "--opt1=42", "sub2", "--sub-opt3=b",
                     "--sub-opt1=13", "--sub-opt2=abc"},
                    llvm::errs()),
              IsSuccess(Eq(ParseResult::Success)));
  EXPECT_THAT(integer_option, Eq(42));
  EXPECT_THAT(string_option, StrEq("xyz"));
  EXPECT_THAT(subcommand, Eq(TestSubcommand::Sub2));
  EXPECT_THAT(integer_sub_option, Eq(13));
  EXPECT_THAT(string_sub_option, StrEq("abc"));
  EXPECT_THAT(enum_sub_option, Eq(TestEnum::Val2));
}

TEST(ArgParserTest, HelpAndVersion) {
  bool flag = false;
  int storage = -1;
  llvm::StringRef string = "";
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(  // Force line break.
        args, s,
        {
            .name = "test",
            .version = "Test Command -- version 1.2.3",
            .build_info = R"""(
Build timestamp: )""" __TIMESTAMP__ R"""(
Build config: test-config-info
)""",
            .help = R"""(
A test command.

Lots more information about the test command.
)""",
            .help_epilogue = R"""(
Closing remarks.
)""",
        },
        [&](auto& b) {
          b.AddFlag(
              {
                  .name = "flag",
                  .short_name = "f",
                  .help = R"""(
A boolean flag.
)""",
              },
              [&](auto& arg_b) { arg_b.Set(&flag); });
          b.AddFlag(
              {
                  .name = "hidden_flag",
                  .help = R"""(
A *hidden* boolean flag.
)""",
              },
              [&](auto& arg_b) {
                arg_b.HelpHidden(true);
                arg_b.Set(&flag);
              });
          b.AddSubcommand(
              {
                  .name = "edit",
                  .help = R"""(
Edit the widget.

This will take the provided widgets and edit them.
)""",
                  .help_epilogue = R"""(
That's all.
)""",
              },
              [&](auto& sub_b) {
                sub_b.AddIntegerOption(
                    {
                        .name = "option",
                        .short_name = "o",
                        .help = R"""(
An integer option.
)""",
                    },
                    [&](auto& arg_b) { arg_b.Set(&storage); });
                sub_b.Do([] {});
              });
          b.AddSubcommand(
              {
                  .name = "run",
                  .help = R"""(
Run wombats across the screen.

This will cause several wombats to run across your screen.
)""",
                  .help_epilogue = R"""(
Or it won't, who knows.
)""",
              },
              [&](auto& sub_b) {
                sub_b.AddStringOption(
                    {
                        .name = "option",
                        .short_name = "o",
                        .help = R"""(
A string option.
)""",
                    },
                    [&](auto& arg_b) { arg_b.Set(&string); });
                sub_b.AddOneOfOption(
                    {
                        .name = "one-of-option",
                        .help = R"""(
A one-of option.
)""",
                    },
                    [&](auto& arg_b) {
                      arg_b.SetOneOf(
                          {
                              arg_b.OneOfValue("x", 1),
                              arg_b.OneOfValue("y", 2),
                              arg_b.OneOfValue("z", 3),
                          },
                          &storage);
                    });
                sub_b.Do([] {});
              });
          b.AddSubcommand(
              {
                  .name = "hidden",
                  .help = R"""(
A hidden subcommand.
)""",
              },
              [&](auto& sub_b) {
                sub_b.HelpHidden(true);
                sub_b.Do([] {});
              });
          b.RequiresSubcommand();
        });
  };

  RawStringOstream os;

  EXPECT_THAT(parse({"--flag", "--help"}, os),
              IsSuccess(Eq(ParseResult::MetaSuccess)));
  std::string help_flag_output = os.TakeStr();
  EXPECT_THAT(help_flag_output, StrEq(llvm::StringRef(R"""(
Test Command -- version 1.2.3

A test command.

Lots more information about the test command.

Build info:
  Build timestamp: )""" __TIMESTAMP__ R"""(
  Build config: test-config-info

Usage:
  test [-f] edit [--option=...]
  test [-f] run [OPTIONS]

Subcommands:
  edit
          Edit the widget.

          This will take the provided widgets and edit them.

  run
          Run wombats across the screen.

          This will cause several wombats to run across your screen.

  help
          Prints help information for the command, including a description, command line usage, and details of each subcommand and option that can be provided.

  version
          Prints the version of this command.

Command options:
  -f, --flag
          A boolean flag.

Closing remarks.

)""")
                                          .ltrim('\n')));
  EXPECT_THAT(parse({"help"}, os), IsSuccess(Eq(ParseResult::MetaSuccess)));
  EXPECT_THAT(os.TakeStr(), StrEq(help_flag_output));

  EXPECT_THAT(parse({"--version"}, os),
              IsSuccess(Eq(ParseResult::MetaSuccess)));
  std::string version_flag_output = os.TakeStr();
  EXPECT_THAT(version_flag_output, StrEq(llvm::StringRef(R"""(
Test Command -- version 1.2.3

Build timestamp: )""" __TIMESTAMP__ R"""(
Build config: test-config-info
)""")
                                             .ltrim('\n')));
  EXPECT_THAT(parse({"version"}, os), IsSuccess(Eq(ParseResult::MetaSuccess)));
  EXPECT_THAT(os.TakeStr(), StrEq(version_flag_output));

  EXPECT_THAT(parse({"--flag", "edit", "--option=42", "--help"}, os),
              IsSuccess(Eq(ParseResult::MetaSuccess)));
  std::string edit_help_output = os.TakeStr();
  EXPECT_THAT(edit_help_output, StrEq(llvm::StringRef(R"""(
Edit the widget.

This will take the provided widgets and edit them.

Subcommand `edit` usage:
  test [-f] edit [--option=...]

Subcommand `edit` options:
  -o, --option=...
          An integer option.

      --help[=(full|short)]
          Prints help information for the subcommand, including a description, command line usage, and details of each option that can be provided.

          Possible values:
          - full (default)
          - short

That's all.

)""")
                                          .ltrim('\n')));

  EXPECT_THAT(parse({"help", "edit"}, os),
              IsSuccess(Eq(ParseResult::MetaSuccess)));
  EXPECT_THAT(os.TakeStr(), StrEq(edit_help_output));

  EXPECT_THAT(parse({"--flag", "run", "--option=abc", "--help"}, os),
              IsSuccess(Eq(ParseResult::MetaSuccess)));
  std::string run_help_output = os.TakeStr();
  EXPECT_THAT(run_help_output, StrEq(llvm::StringRef(R"""(
Run wombats across the screen.

This will cause several wombats to run across your screen.

Subcommand `run` usage:
  test [-f] run [OPTIONS]

Subcommand `run` options:
  -o, --option=...
          A string option.

      --one-of-option=(x|y|z)
          A one-of option.

          Possible values:
          - x
          - y
          - z

      --help[=(full|short)]
          Prints help information for the subcommand, including a description, command line usage, and details of each option that can be provided.

          Possible values:
          - full (default)
          - short

Or it won't, who knows.

)""")
                                         .ltrim('\n')));

  EXPECT_THAT(parse({"help", "run"}, os),
              IsSuccess(Eq(ParseResult::MetaSuccess)));
  EXPECT_THAT(os.TakeStr(), StrEq(run_help_output));
}

TEST(ArgParserTest, HelpMarkdownLike) {
  bool flag = false;
  auto parse = [&](llvm::ArrayRef<llvm::StringRef> args, llvm::raw_ostream& s) {
    return Parse(  // Force line break.
        args, s, {.name = "test"}, [&](auto& b) {
          b.AddFlag(
              {
                  .name = "flag",
                  .help = R"""(
A boolean flag.

    Preformatted
        code....
        ........

But
  here
    lines
      are
        collapsed.

```
x
 y
  z
```
)""",
              },
              [&](auto& arg_b) { arg_b.Set(&flag); });
          b.Do([] {});
        });
  };

  RawStringOstream os;
  EXPECT_THAT(parse({"--help"}, os), IsSuccess(Eq(ParseResult::MetaSuccess)));
  EXPECT_THAT(os.TakeStr(), StrEq(llvm::StringRef(R"""(
Usage:
  test [--flag]

Options:
      --flag
          A boolean flag.

              Preformatted
                  code....
                  ........

          But here lines are collapsed.

          ```
          x
           y
            z
          ```

      --help[=(full|short)]
          Prints help information for the command, including a description, command line usage, and details of each option that can be provided.

          Possible values:
          - full (default)
          - short

)""")
                                      .ltrim('\n')));
}

}  // namespace
}  // namespace Carbon::CommandLine
