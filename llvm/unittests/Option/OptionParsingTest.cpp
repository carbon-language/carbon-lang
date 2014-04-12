//===- unittest/Support/OptionParsingTest.cpp - OptTable tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::opt;

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR) OPT_##ID,
#include "Opts.inc"
  LastOption
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

enum OptionFlags {
  OptFlag1 = (1 << 4),
  OptFlag2 = (1 << 5),
  OptFlag3 = (1 << 6)
};

static const OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, Option::KIND##Class, PARAM, \
    FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS },
#include "Opts.inc"
#undef OPTION
};

namespace {
class TestOptTable : public OptTable {
public:
  TestOptTable(bool IgnoreCase = false)
    : OptTable(InfoTable, array_lengthof(InfoTable), IgnoreCase) {}
};
}

const char *Args[] = {
  "-A",
  "-Bhi",
  "--C=desu",
  "-C", "bye",
  "-D,adena",
  "-E", "apple", "bloom",
  "-Fblarg",
  "-F", "42",
  "-Gchuu", "2"
  };

TEST(Option, OptionParsing) {
  TestOptTable T;
  unsigned MAI, MAC;
  std::unique_ptr<InputArgList> AL(
      T.ParseArgs(std::begin(Args), std::end(Args), MAI, MAC));

  // Check they all exist.
  EXPECT_TRUE(AL->hasArg(OPT_A));
  EXPECT_TRUE(AL->hasArg(OPT_B));
  EXPECT_TRUE(AL->hasArg(OPT_C));
  EXPECT_TRUE(AL->hasArg(OPT_D));
  EXPECT_TRUE(AL->hasArg(OPT_E));
  EXPECT_TRUE(AL->hasArg(OPT_F));
  EXPECT_TRUE(AL->hasArg(OPT_G));

  // Check the values.
  EXPECT_EQ(AL->getLastArgValue(OPT_B), "hi");
  EXPECT_EQ(AL->getLastArgValue(OPT_C), "bye");
  EXPECT_EQ(AL->getLastArgValue(OPT_D), "adena");
  std::vector<std::string> Es = AL->getAllArgValues(OPT_E);
  EXPECT_EQ(Es[0], "apple");
  EXPECT_EQ(Es[1], "bloom");
  EXPECT_EQ(AL->getLastArgValue(OPT_F), "42");
  std::vector<std::string> Gs = AL->getAllArgValues(OPT_G);
  EXPECT_EQ(Gs[0], "chuu");
  EXPECT_EQ(Gs[1], "2");

  // Check the help text.
  std::string Help;
  raw_string_ostream RSO(Help);
  T.PrintHelp(RSO, "test", "title!");
  EXPECT_NE(Help.find("-A"), std::string::npos);

  // Test aliases.
  arg_iterator Cs = AL->filtered_begin(OPT_C);
  ASSERT_NE(Cs, AL->filtered_end());
  EXPECT_EQ(StringRef((*Cs)->getValue()), "desu");
  ArgStringList ASL;
  (*Cs)->render(*AL, ASL);
  ASSERT_EQ(ASL.size(), 2u);
  EXPECT_EQ(StringRef(ASL[0]), "-C");
  EXPECT_EQ(StringRef(ASL[1]), "desu");
}

TEST(Option, ParseWithFlagExclusions) {
  TestOptTable T;
  unsigned MAI, MAC;
  std::unique_ptr<InputArgList> AL;

  // Exclude flag3 to avoid parsing as OPT_SLASH_C.
  AL.reset(T.ParseArgs(std::begin(Args), std::end(Args), MAI, MAC,
                       /*FlagsToInclude=*/0,
                       /*FlagsToExclude=*/OptFlag3));
  EXPECT_TRUE(AL->hasArg(OPT_A));
  EXPECT_TRUE(AL->hasArg(OPT_C));
  EXPECT_FALSE(AL->hasArg(OPT_SLASH_C));

  // Exclude flag1 to avoid parsing as OPT_C.
  AL.reset(T.ParseArgs(std::begin(Args), std::end(Args), MAI, MAC,
                       /*FlagsToInclude=*/0,
                       /*FlagsToExclude=*/OptFlag1));
  EXPECT_TRUE(AL->hasArg(OPT_B));
  EXPECT_FALSE(AL->hasArg(OPT_C));
  EXPECT_TRUE(AL->hasArg(OPT_SLASH_C));

  const char *NewArgs[] = { "/C", "foo", "--C=bar" };
  AL.reset(T.ParseArgs(std::begin(NewArgs), std::end(NewArgs), MAI, MAC));
  EXPECT_TRUE(AL->hasArg(OPT_SLASH_C));
  EXPECT_TRUE(AL->hasArg(OPT_C));
  EXPECT_EQ(AL->getLastArgValue(OPT_SLASH_C), "foo");
  EXPECT_EQ(AL->getLastArgValue(OPT_C), "bar");
}

TEST(Option, ParseAliasInGroup) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-I" };
  std::unique_ptr<InputArgList> AL(
      T.ParseArgs(std::begin(MyArgs), std::end(MyArgs), MAI, MAC));
  EXPECT_TRUE(AL->hasArg(OPT_H));
}

TEST(Option, AliasArgs) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-J", "-Joo" };
  std::unique_ptr<InputArgList> AL(
      T.ParseArgs(std::begin(MyArgs), std::end(MyArgs), MAI, MAC));
  EXPECT_TRUE(AL->hasArg(OPT_B));
  EXPECT_EQ(AL->getAllArgValues(OPT_B)[0], "foo");
  EXPECT_EQ(AL->getAllArgValues(OPT_B)[1], "bar");
}

TEST(Option, IgnoreCase) {
  TestOptTable T(true);
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-a", "-joo" };
  std::unique_ptr<InputArgList> AL(
      T.ParseArgs(std::begin(MyArgs), std::end(MyArgs), MAI, MAC));
  EXPECT_TRUE(AL->hasArg(OPT_A));
  EXPECT_TRUE(AL->hasArg(OPT_B));
}

TEST(Option, DoNotIgnoreCase) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-a", "-joo" };
  std::unique_ptr<InputArgList> AL(
      T.ParseArgs(std::begin(MyArgs), std::end(MyArgs), MAI, MAC));
  EXPECT_FALSE(AL->hasArg(OPT_A));
  EXPECT_FALSE(AL->hasArg(OPT_B));
}

TEST(Option, SlurpEmpty) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurp" };
  std::unique_ptr<InputArgList> AL(
      T.ParseArgs(std::begin(MyArgs), std::end(MyArgs), MAI, MAC));
  EXPECT_TRUE(AL->hasArg(OPT_A));
  EXPECT_TRUE(AL->hasArg(OPT_Slurp));
  EXPECT_EQ(AL->getAllArgValues(OPT_Slurp).size(), 0U);
}

TEST(Option, Slurp) {
  TestOptTable T;
  unsigned MAI, MAC;

  const char *MyArgs[] = { "-A", "-slurp", "-B", "--", "foo" };
  std::unique_ptr<InputArgList> AL(
      T.ParseArgs(std::begin(MyArgs), std::end(MyArgs), MAI, MAC));
  EXPECT_EQ(AL->size(), 2U);
  EXPECT_TRUE(AL->hasArg(OPT_A));
  EXPECT_FALSE(AL->hasArg(OPT_B));
  EXPECT_TRUE(AL->hasArg(OPT_Slurp));
  EXPECT_EQ(AL->getAllArgValues(OPT_Slurp).size(), 3U);
  EXPECT_EQ(AL->getAllArgValues(OPT_Slurp)[0], "-B");
  EXPECT_EQ(AL->getAllArgValues(OPT_Slurp)[1], "--");
  EXPECT_EQ(AL->getAllArgValues(OPT_Slurp)[2], "foo");
}
