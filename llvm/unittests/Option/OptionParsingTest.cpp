//===- unittest/Support/OptionParsingTest.cpp - OptTable tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::opt;

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
              HELPTEXT, METAVAR) OPT_##ID,
#include "Opts.inc"
  LastOption
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

static const OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, Option::KIND##Class, PARAM, \
    FLAGS, OPT_##GROUP, OPT_##ALIAS },
#include "Opts.inc"
#undef OPTION
};

namespace {
class TestOptTable : public OptTable {
public:
  TestOptTable()
    : OptTable(InfoTable, sizeof(InfoTable) / sizeof(InfoTable[0])) {}
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

TEST(Support, OptionParsing) {
  TestOptTable T;
  unsigned MAI, MAC;
  OwningPtr<InputArgList>
    AL(T.ParseArgs(Args,
                   Args + (sizeof(Args) / sizeof(Args[0])),
                   MAI,
                   MAC));

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
