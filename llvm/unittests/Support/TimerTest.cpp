//===- unittests/TimerTest.cpp - Timer tests ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"
#include "llvm/Support/thread.h"
#include "gtest/gtest.h"
#include <chrono>

using namespace llvm;

namespace {

TEST(Timer, Additivity) {
  Timer T1("T1");

  EXPECT_TRUE(T1.isInitialized());

  T1.startTimer();
  T1.stopTimer();
  auto TR1 = T1.getTotalTime();

  T1.startTimer();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  T1.stopTimer();
  auto TR2 = T1.getTotalTime();

  EXPECT_TRUE(TR1 < TR2);
}

TEST(Timer, CheckIfTriggered) {
  Timer T1("T1");

  EXPECT_FALSE(T1.hasTriggered());
  T1.startTimer();
  EXPECT_TRUE(T1.hasTriggered());
  T1.stopTimer();
  EXPECT_TRUE(T1.hasTriggered());

  T1.clear();
  EXPECT_FALSE(T1.hasTriggered());
}

} // end anon namespace
