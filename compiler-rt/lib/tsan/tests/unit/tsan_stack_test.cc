//===-- tsan_stack_test.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_sync.h"
#include "tsan_rtl.h"
#include "gtest/gtest.h"
#include <string.h>

namespace __tsan {

static void TestStackTrace(StackTrace *trace) {
  ThreadState thr(0, 0, 0, 0, 0, 0, 0, 0);

  trace->ObtainCurrent(&thr, 0);
  EXPECT_EQ(trace->Size(), (uptr)0);

  trace->ObtainCurrent(&thr, 42);
  EXPECT_EQ(trace->Size(), (uptr)1);
  EXPECT_EQ(trace->Get(0), (uptr)42);

  *thr.shadow_stack_pos++ = 100;
  *thr.shadow_stack_pos++ = 101;
  trace->ObtainCurrent(&thr, 0);
  EXPECT_EQ(trace->Size(), (uptr)2);
  EXPECT_EQ(trace->Get(0), (uptr)100);
  EXPECT_EQ(trace->Get(1), (uptr)101);

  trace->ObtainCurrent(&thr, 42);
  EXPECT_EQ(trace->Size(), (uptr)3);
  EXPECT_EQ(trace->Get(0), (uptr)100);
  EXPECT_EQ(trace->Get(1), (uptr)101);
  EXPECT_EQ(trace->Get(2), (uptr)42);
}

TEST(StackTrace, Basic) {
  ScopedInRtl in_rtl;
  StackTrace trace;
  TestStackTrace(&trace);
}

TEST(StackTrace, StaticBasic) {
  ScopedInRtl in_rtl;
  uptr buf[10];
  StackTrace trace1(buf, 10);
  TestStackTrace(&trace1);
  StackTrace trace2(buf, 3);
  TestStackTrace(&trace2);
}

TEST(StackTrace, StaticTrim) {
  ScopedInRtl in_rtl;
  uptr buf[2];
  StackTrace trace(buf, 2);
  ThreadState thr(0, 0, 0, 0, 0, 0, 0, 0);

  *thr.shadow_stack_pos++ = 100;
  *thr.shadow_stack_pos++ = 101;
  *thr.shadow_stack_pos++ = 102;
  trace.ObtainCurrent(&thr, 0);
  EXPECT_EQ(trace.Size(), (uptr)2);
  EXPECT_EQ(trace.Get(0), (uptr)101);
  EXPECT_EQ(trace.Get(1), (uptr)102);

  trace.ObtainCurrent(&thr, 42);
  EXPECT_EQ(trace.Size(), (uptr)2);
  EXPECT_EQ(trace.Get(0), (uptr)102);
  EXPECT_EQ(trace.Get(1), (uptr)42);
}


}  // namespace __tsan
