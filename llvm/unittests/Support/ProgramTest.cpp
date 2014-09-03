//===- unittest/Support/ProgramTest.cpp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "gtest/gtest.h"
#include <stdlib.h>
#if defined(__APPLE__)
# include <crt_externs.h>
#elif !defined(_MSC_VER)
// Forward declare environ in case it's not provided by stdlib.h.
extern char **environ;
#endif

#if defined(LLVM_ON_UNIX)
#include <unistd.h>
void sleep_for(unsigned int seconds) {
  sleep(seconds);
}
#elif defined(LLVM_ON_WIN32)
#include <windows.h>
void sleep_for(unsigned int seconds) {
  Sleep(seconds * 1000);
}
#else
#error sleep_for is not implemented on your platform.
#endif

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }
// From TestMain.cpp.
extern const char *TestMainArgv0;

namespace {

using namespace llvm;
using namespace sys;

static cl::opt<std::string>
ProgramTestStringArg1("program-test-string-arg1");
static cl::opt<std::string>
ProgramTestStringArg2("program-test-string-arg2");

static void CopyEnvironment(std::vector<const char *> &out) {
#ifdef __APPLE__
  char **envp = *_NSGetEnviron();
#else
  // environ seems to work for Windows and most other Unices.
  char **envp = environ;
#endif
  while (*envp != nullptr) {
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

  std::string my_exe =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  const char *argv[] = {
    my_exe.c_str(),
    "--gtest_filter=ProgramTest.CreateProcessTrailingSlash",
    "-program-test-string-arg1", "has\\\\ trailing\\",
    "-program-test-string-arg2", "has\\\\ trailing\\",
    nullptr
  };

  // Add LLVM_PROGRAM_TEST_CHILD to the environment of the child.
  std::vector<const char *> envp;
  CopyEnvironment(envp);
  envp.push_back("LLVM_PROGRAM_TEST_CHILD=1");
  envp.push_back(nullptr);

  std::string error;
  bool ExecutionFailed;
  // Redirect stdout and stdin to NUL, but let stderr through.
#ifdef LLVM_ON_WIN32
  StringRef nul("NUL");
#else
  StringRef nul("/dev/null");
#endif
  const StringRef *redirects[] = { &nul, &nul, nullptr };
  int rc = ExecuteAndWait(my_exe, argv, &envp[0], redirects,
                          /*secondsToWait=*/ 10, /*memoryLimit=*/ 0, &error,
                          &ExecutionFailed);
  EXPECT_FALSE(ExecutionFailed) << error;
  EXPECT_EQ(0, rc);
}

TEST(ProgramTest, TestExecuteNoWait) {
  using namespace llvm::sys;

  if (getenv("LLVM_PROGRAM_TEST_EXECUTE_NO_WAIT")) {
    sleep_for(/*seconds*/ 1);
    exit(0);
  }

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  const char *argv[] = {
    Executable.c_str(),
    "--gtest_filter=ProgramTest.TestExecuteNoWait",
    nullptr
  };

  // Add LLVM_PROGRAM_TEST_EXECUTE_NO_WAIT to the environment of the child.
  std::vector<const char *> envp;
  CopyEnvironment(envp);
  envp.push_back("LLVM_PROGRAM_TEST_EXECUTE_NO_WAIT=1");
  envp.push_back(nullptr);

  std::string Error;
  bool ExecutionFailed;
  ProcessInfo PI1 = ExecuteNoWait(Executable, argv, &envp[0], nullptr, 0,
                                  &Error, &ExecutionFailed);
  ASSERT_FALSE(ExecutionFailed) << Error;
  ASSERT_NE(PI1.Pid, 0) << "Invalid process id";

  unsigned LoopCount = 0;

  // Test that Wait() with WaitUntilTerminates=true works. In this case,
  // LoopCount should only be incremented once.
  while (true) {
    ++LoopCount;
    ProcessInfo WaitResult = Wait(PI1, 0, true, &Error);
    ASSERT_TRUE(Error.empty());
    if (WaitResult.Pid == PI1.Pid)
      break;
  }

  EXPECT_EQ(LoopCount, 1u) << "LoopCount should be 1";

  ProcessInfo PI2 = ExecuteNoWait(Executable, argv, &envp[0], nullptr, 0,
                                  &Error, &ExecutionFailed);
  ASSERT_FALSE(ExecutionFailed) << Error;
  ASSERT_NE(PI2.Pid, 0) << "Invalid process id";

  // Test that Wait() with SecondsToWait=0 performs a non-blocking wait. In this
  // cse, LoopCount should be greater than 1 (more than one increment occurs).
  while (true) {
    ++LoopCount;
    ProcessInfo WaitResult = Wait(PI2, 0, false, &Error);
    ASSERT_TRUE(Error.empty());
    if (WaitResult.Pid == PI2.Pid)
      break;
  }

  ASSERT_GT(LoopCount, 1u) << "LoopCount should be >1";
}

TEST(ProgramTest, TestExecuteAndWaitTimeout) {
  using namespace llvm::sys;

  if (getenv("LLVM_PROGRAM_TEST_TIMEOUT")) {
    sleep_for(/*seconds*/ 10);
    exit(0);
  }

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &ProgramTestStringArg1);
  const char *argv[] = {
    Executable.c_str(),
    "--gtest_filter=ProgramTest.TestExecuteAndWaitTimeout",
    nullptr
  };

  // Add LLVM_PROGRAM_TEST_TIMEOUT to the environment of the child.
  std::vector<const char *> envp;
  CopyEnvironment(envp);
  envp.push_back("LLVM_PROGRAM_TEST_TIMEOUT=1");
  envp.push_back(nullptr);

  std::string Error;
  bool ExecutionFailed;
  int RetCode =
      ExecuteAndWait(Executable, argv, &envp[0], nullptr, /*secondsToWait=*/1, 0,
                     &Error, &ExecutionFailed);
  ASSERT_EQ(-2, RetCode);
}

TEST(ProgramTest, TestExecuteNegative) {
  std::string Executable = "i_dont_exist";
  const char *argv[] = { Executable.c_str(), nullptr };

  {
    std::string Error;
    bool ExecutionFailed;
    int RetCode = ExecuteAndWait(Executable, argv, nullptr, nullptr, 0, 0,
                                 &Error, &ExecutionFailed);
    ASSERT_TRUE(RetCode < 0) << "On error ExecuteAndWait should return 0 or "
                                "positive value indicating the result code";
    ASSERT_TRUE(ExecutionFailed);
    ASSERT_FALSE(Error.empty());
  }

  {
    std::string Error;
    bool ExecutionFailed;
    ProcessInfo PI = ExecuteNoWait(Executable, argv, nullptr, nullptr, 0,
                                   &Error, &ExecutionFailed);
    ASSERT_EQ(PI.Pid, 0)
        << "On error ExecuteNoWait should return an invalid ProcessInfo";
    ASSERT_TRUE(ExecutionFailed);
    ASSERT_FALSE(Error.empty());
  }

}

#ifdef LLVM_ON_WIN32
const char utf16le_text[] =
    "\x6c\x00\x69\x00\x6e\x00\x67\x00\xfc\x00\x69\x00\xe7\x00\x61\x00";
const char utf16be_text[] =
    "\x00\x6c\x00\x69\x00\x6e\x00\x67\x00\xfc\x00\x69\x00\xe7\x00\x61";
#endif
const char utf8_text[] = "\x6c\x69\x6e\x67\xc3\xbc\x69\xc3\xa7\x61";

TEST(ProgramTest, TestWriteWithSystemEncoding) {
  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(fs::createUniqueDirectory("program-test", TestDirectory));
  errs() << "Test Directory: " << TestDirectory << '\n';
  errs().flush();
  SmallString<128> file_pathname(TestDirectory);
  path::append(file_pathname, "international-file.txt");
  // Only on Windows we should encode in UTF16. For other systems, use UTF8
  ASSERT_NO_ERROR(sys::writeFileWithEncoding(file_pathname.c_str(), utf8_text,
                                             sys::WEM_UTF16));
  int fd = 0;
  ASSERT_NO_ERROR(fs::openFileForRead(file_pathname.c_str(), fd));
#if defined(LLVM_ON_WIN32)
  char buf[18];
  ASSERT_EQ(::read(fd, buf, 18), 18);
  if (strncmp(buf, "\xfe\xff", 2) == 0) { // UTF16-BE
    ASSERT_EQ(strncmp(&buf[2], utf16be_text, 16), 0);
  } else if (strncmp(buf, "\xff\xfe", 2) == 0) { // UTF16-LE
    ASSERT_EQ(strncmp(&buf[2], utf16le_text, 16), 0);
  } else {
    FAIL() << "Invalid BOM in UTF-16 file";
  }
#else
  char buf[10];
  ASSERT_EQ(::read(fd, buf, 10), 10);
  ASSERT_EQ(strncmp(buf, utf8_text, 10), 0);
#endif
  ::close(fd);
  ASSERT_NO_ERROR(fs::remove(file_pathname.str()));
  ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
}

} // end anonymous namespace
