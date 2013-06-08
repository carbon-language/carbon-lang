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
#include "windows.h"
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

} // end anonymous namespace
