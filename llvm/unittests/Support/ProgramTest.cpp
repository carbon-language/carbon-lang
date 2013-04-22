//===- unittest/Support/ProgramTest.cpp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "gtest/gtest.h"

#include <stdlib.h>

namespace {

using namespace llvm;
using namespace sys;

static cl::opt<std::string>
ProgramTestStringArg1("program-test-string-arg1");
static cl::opt<std::string>
ProgramTestStringArg2("program-test-string-arg2");

static void CopyEnvironment(std::vector<const char *> out) {
  // environ appears to be pretty portable.
  char **envp = environ;
  while (*envp != 0) {
    out.push_back(*envp);
    ++envp;
  }
}

TEST(ProgramTest, CreateProcessTrailingSlash) {
  if (getenv("LLVM_PROGRAM_TEST_CHILD")) {
    if (ProgramTestStringArg1 == "has\\\\ trailing\\" &&
        ProgramTestStringArg2 == "has\\\\ trailing\\") {
      exit(0);  // Success!  The arguments were passed and parsed.
    }
    exit(1);
  }

  // FIXME: Hardcoding argv0 here since I don't know a good cross-platform way
  // to get it.  Maybe ParseCommandLineOptions() should save it?
  Path my_exe = Path::GetMainExecutable("SupportTests", &ProgramTestStringArg1);
  const char *argv[] = {
    my_exe.c_str(),
    "--gtest_filter=ProgramTest.CreateProcessTrailingSlashChild",
    "-program-test-string-arg1", "has\\\\ trailing\\",
    "-program-test-string-arg2", "has\\\\ trailing\\",
    0
  };

  // Add LLVM_PROGRAM_TEST_CHILD to the environment of the child.
  std::vector<const char *> envp;
  CopyEnvironment(envp);
  envp.push_back("LLVM_PROGRAM_TEST_CHILD=1");
  envp.push_back(0);

  std::string error;
  bool ExecutionFailed;
  // Redirect stdout and stdin to NUL, but let stderr through.
#ifdef LLVM_ON_WIN32
  Path nul("NUL");
#else
  Path nul("/dev/null");
#endif
  const Path *redirects[] = { &nul, &nul, 0 };
  int rc = Program::ExecuteAndWait(my_exe, argv, &envp[0], redirects,
                                   /*secondsToWait=*/10, /*memoryLimit=*/0,
                                   &error, &ExecutionFailed);
  EXPECT_FALSE(ExecutionFailed) << error;
  EXPECT_EQ(0, rc);
}

} // end anonymous namespace
