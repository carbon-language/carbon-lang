//===- llvm/unittest/Support/CommandLineTest.cpp - CommandLine tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"

#include "gtest/gtest.h"

#include <string>
#include <stdlib.h>

using namespace llvm;

namespace {

class TempEnvVar {
 public:
  TempEnvVar(const char *name, const char *value)
      : name(name) {
    const char *old_value = getenv(name);
    EXPECT_EQ(NULL, old_value) << old_value;
    setenv(name, value, true);
  }

  ~TempEnvVar() {
    unsetenv(name);
  }

 private:
  const char *const name;
};

const char test_env_var[] = "LLVM_TEST_COMMAND_LINE_FLAGS";

cl::opt<std::string> EnvironmentTestOption("env-test-opt");
TEST(CommandLineTest, ParseEnvironment) {
  TempEnvVar TEV(test_env_var, "-env-test-opt=hello");
  EXPECT_EQ("", EnvironmentTestOption);
  cl::ParseEnvironmentOptions("CommandLineTest", test_env_var);
  EXPECT_EQ("hello", EnvironmentTestOption);
}

}  // anonymous namespace
