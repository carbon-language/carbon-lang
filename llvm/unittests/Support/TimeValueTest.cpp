//===- llvm/unittest/Support/TimeValueTest.cpp - Time Value tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Support/TimeValue.h"
#include <time.h>

using namespace llvm;
namespace {

TEST(TimeValue, time_t) {
  sys::TimeValue now = sys::TimeValue::now();
  time_t now_t = time(NULL);
  EXPECT_TRUE(std::abs(static_cast<long>(now_t - now.toEpochTime())) < 2);
}

TEST(TimeValue, Win32FILETIME) {
  uint64_t epoch_as_filetime = 0x19DB1DED53E8000ULL;
  uint32_t ns = 765432100;
  sys::TimeValue epoch;

  // FILETIME has 100ns of intervals.
  uint64_t ft1970 = epoch_as_filetime + ns / 100;
  epoch.fromWin32Time(ft1970);

  // The "seconds" part in Posix time may be expected as zero.
  EXPECT_EQ(0u, epoch.toEpochTime());
  EXPECT_EQ(ns, static_cast<uint32_t>(epoch.nanoseconds()));

  // Confirm it reversible.
  EXPECT_EQ(ft1970, epoch.toWin32Time());
}

}
