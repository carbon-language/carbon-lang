//===- unittest/Support/ProcessTest.cpp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Process.h"
#include "gtest/gtest.h"

#ifdef LLVM_ON_WIN32
#include <windows.h>
#endif

namespace {

using namespace llvm;
using namespace sys;

TEST(ProcessTest, SelfProcess) {
  EXPECT_TRUE(process::get_self());
  EXPECT_EQ(process::get_self(), process::get_self());

#if defined(LLVM_ON_UNIX)
  EXPECT_EQ(getpid(), process::get_self()->get_id());
#elif defined(LLVM_ON_WIN32)
  EXPECT_EQ(GetCurrentProcessId(), process::get_self()->get_id());
#endif

  EXPECT_LT(1u, process::get_self()->page_size());

  EXPECT_LT(TimeValue::MinTime, process::get_self()->get_user_time());
  EXPECT_GT(TimeValue::MaxTime, process::get_self()->get_user_time());
  EXPECT_LT(TimeValue::MinTime, process::get_self()->get_system_time());
  EXPECT_GT(TimeValue::MaxTime, process::get_self()->get_system_time());
  EXPECT_LT(TimeValue::MinTime, process::get_self()->get_wall_time());
  EXPECT_GT(TimeValue::MaxTime, process::get_self()->get_wall_time());
}

TEST(ProcessTest, GetRandomNumberTest) {
  const unsigned r1 = Process::GetRandomNumber();
  const unsigned r2 = Process::GetRandomNumber();
  // It should be extremely unlikely that both r1 and r2 are 0.
  EXPECT_NE((r1 | r2), 0u);
}

#ifdef _MSC_VER
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

#if HAVE_SETENV || _MSC_VER
TEST(ProcessTest, Basic) {
  setenv("__LLVM_TEST_ENVIRON_VAR__", "abc", true);
  Optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.hasValue());
  EXPECT_STREQ("abc", val->c_str());
}

TEST(ProcessTest, None) {
  Optional<std::string> val(
      Process::GetEnv("__LLVM_TEST_ENVIRON_NO_SUCH_VAR__"));
  EXPECT_FALSE(val.hasValue());
}
#endif

#ifdef LLVM_ON_WIN32
TEST(ProcessTest, Wchar) {
  SetEnvironmentVariableW(L"__LLVM_TEST_ENVIRON_VAR__", L"abcdefghijklmnopqrs");
  Optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.hasValue());
  EXPECT_STREQ("abcdefghijklmnopqrs", val->c_str());
}
#endif

} // end anonymous namespace
