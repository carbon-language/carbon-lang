//===- llvm/unittest/Support/CommandLineTest.cpp - CommandLine tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Config/config.h"
#include "gtest/gtest.h"
#include <stdlib.h>
#include <string>

using namespace llvm;

namespace {

class TempEnvVar {
 public:
  TempEnvVar(const char *name, const char *value)
      : name(name) {
    const char *old_value = getenv(name);
    EXPECT_EQ(NULL, old_value) << old_value;
#if HAVE_SETENV
    setenv(name, value, true);
#else
#   define SKIP_ENVIRONMENT_TESTS
#endif
  }

  ~TempEnvVar() {
#if HAVE_SETENV
    // Assume setenv and unsetenv come together.
    unsetenv(name);
#endif
  }

 private:
  const char *const name;
};

cl::OptionCategory TestCategory("Test Options", "Description");
cl::opt<int> TestOption("test-option", cl::desc("old description"));
TEST(CommandLineTest, ModifyExisitingOption) {
  const char Description[] = "New description";
  const char ArgString[] = "new-test-option";
  const char ValueString[] = "Integer";

  StringMap<cl::Option*> Map;
  cl::getRegisteredOptions(Map);

  ASSERT_TRUE(Map.count("test-option") == 1) <<
    "Could not find option in map.";

  cl::Option *Retrieved = Map["test-option"];
  ASSERT_EQ(&TestOption, Retrieved) << "Retrieved wrong option.";

  ASSERT_EQ(&cl::GeneralCategory,Retrieved->Category) <<
    "Incorrect default option category.";

  Retrieved->setCategory(TestCategory);
  ASSERT_EQ(&TestCategory,Retrieved->Category) <<
    "Failed to modify option's option category.";

  Retrieved->setDescription(Description);
  ASSERT_STREQ(Retrieved->HelpStr, Description) <<
    "Changing option description failed.";

  Retrieved->setArgStr(ArgString);
  ASSERT_STREQ(ArgString, Retrieved->ArgStr) <<
    "Failed to modify option's Argument string.";

  Retrieved->setValueStr(ValueString);
  ASSERT_STREQ(Retrieved->ValueStr, ValueString) <<
    "Failed to modify option's Value string.";

  Retrieved->setHiddenFlag(cl::Hidden);
  ASSERT_EQ(cl::Hidden, TestOption.getOptionHiddenFlag()) <<
    "Failed to modify option's hidden flag.";
}
#ifndef SKIP_ENVIRONMENT_TESTS

const char test_env_var[] = "LLVM_TEST_COMMAND_LINE_FLAGS";

cl::opt<std::string> EnvironmentTestOption("env-test-opt");
TEST(CommandLineTest, ParseEnvironment) {
  TempEnvVar TEV(test_env_var, "-env-test-opt=hello");
  EXPECT_EQ("", EnvironmentTestOption);
  cl::ParseEnvironmentOptions("CommandLineTest", test_env_var);
  EXPECT_EQ("hello", EnvironmentTestOption);
}

// This test used to make valgrind complain
// ("Conditional jump or move depends on uninitialised value(s)")
//
// Warning: Do not run any tests after this one that try to gain access to
// registered command line options because this will likely result in a
// SEGFAULT. This can occur because the cl::opt in the test below is declared
// on the stack which will be destroyed after the test completes but the
// command line system will still hold a pointer to a deallocated cl::Option.
TEST(CommandLineTest, ParseEnvironmentToLocalVar) {
  // Put cl::opt on stack to check for proper initialization of fields.
  cl::opt<std::string> EnvironmentTestOptionLocal("env-test-opt-local");
  TempEnvVar TEV(test_env_var, "-env-test-opt-local=hello-local");
  EXPECT_EQ("", EnvironmentTestOptionLocal);
  cl::ParseEnvironmentOptions("CommandLineTest", test_env_var);
  EXPECT_EQ("hello-local", EnvironmentTestOptionLocal);
}

#endif  // SKIP_ENVIRONMENT_TESTS

TEST(CommandLineTest, UseOptionCategory) {
  cl::opt<int> TestOption2("test-option", cl::cat(TestCategory));

  ASSERT_EQ(&TestCategory,TestOption2.Category) << "Failed to assign Option "
                                                  "Category.";
}

class StrDupSaver : public cl::StringSaver {
  const char *SaveString(const char *Str) LLVM_OVERRIDE {
    return strdup(Str);
  }
};

typedef void ParserFunction(StringRef Source, llvm::cl::StringSaver &Saver,
                            SmallVectorImpl<const char *> &NewArgv);


void testCommandLineTokenizer(ParserFunction *parse, const char *Input,
                              const char *const Output[], size_t OutputSize) {
  SmallVector<const char *, 0> Actual;
  StrDupSaver Saver;
  parse(Input, Saver, Actual);
  EXPECT_EQ(OutputSize, Actual.size());
  for (unsigned I = 0, E = Actual.size(); I != E; ++I) {
    if (I < OutputSize)
      EXPECT_STREQ(Output[I], Actual[I]);
    free(const_cast<char *>(Actual[I]));
  }
}

TEST(CommandLineTest, TokenizeGNUCommandLine) {
  const char *Input = "foo\\ bar \"foo bar\" \'foo bar\' 'foo\\\\bar' "
                      "foo\"bar\"baz C:\\src\\foo.cpp \"C:\\src\\foo.cpp\"";
  const char *const Output[] = { "foo bar", "foo bar", "foo bar", "foo\\bar",
                                 "foobarbaz", "C:\\src\\foo.cpp",
                                 "C:\\src\\foo.cpp" };
  testCommandLineTokenizer(cl::TokenizeGNUCommandLine, Input, Output,
                           array_lengthof(Output));
}

TEST(CommandLineTest, TokenizeWindowsCommandLine) {
  const char *Input = "a\\b c\\\\d e\\\\\"f g\" h\\\"i j\\\\\\\"k \"lmn\" o pqr "
                      "\"st \\\"u\" \\v";
  const char *const Output[] = { "a\\b", "c\\\\d", "e\\f g", "h\"i", "j\\\"k",
                                 "lmn", "o", "pqr", "st \"u", "\\v" };
  testCommandLineTokenizer(cl::TokenizeWindowsCommandLine, Input, Output,
                           array_lengthof(Output));
}

}  // anonymous namespace
