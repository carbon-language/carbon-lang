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

template <typename StackTraceTy>
static void TestStackTrace(StackTraceTy *trace) {
  ThreadState thr(0, 0, 0, 0, 0, 0, 0, 0, 0);
  uptr stack[128];
  thr.shadow_stack = &stack[0];
  thr.shadow_stack_pos = &stack[0];
  thr.shadow_stack_end = &stack[128];

  ObtainCurrentStack(&thr, 0, trace);
  EXPECT_EQ(0U, trace->size);

  ObtainCurrentStack(&thr, 42, trace);
  EXPECT_EQ(1U, trace->size);
  EXPECT_EQ(42U, trace->trace[0]);

  *thr.shadow_stack_pos++ = 100;
  *thr.shadow_stack_pos++ = 101;
  ObtainCurrentStack(&thr, 0, trace);
  EXPECT_EQ(2U, trace->size);
  EXPECT_EQ(100U, trace->trace[0]);
  EXPECT_EQ(101U, trace->trace[1]);

  ObtainCurrentStack(&thr, 42, trace);
  EXPECT_EQ(3U, trace->size);
  EXPECT_EQ(100U, trace->trace[0]);
  EXPECT_EQ(101U, trace->trace[1]);
  EXPECT_EQ(42U, trace->trace[2]);
}

template<typename StackTraceTy>
static void TestTrim(StackTraceTy *trace) {
  ThreadState thr(0, 0, 0, 0, 0, 0, 0, 0, 0);
  const uptr kShadowStackSize = 2 * kStackTraceMax;
  uptr stack[kShadowStackSize];
  thr.shadow_stack = &stack[0];
  thr.shadow_stack_pos = &stack[0];
  thr.shadow_stack_end = &stack[kShadowStackSize];

  for (uptr i = 0; i < kShadowStackSize; ++i)
    *thr.shadow_stack_pos++ = 100 + i;

  ObtainCurrentStack(&thr, 0, trace);
  EXPECT_EQ(kStackTraceMax, trace->size);
  for (uptr i = 0; i < kStackTraceMax; i++) {
    EXPECT_EQ(100 + kStackTraceMax + i, trace->trace[i]);
  }

  ObtainCurrentStack(&thr, 42, trace);
  EXPECT_EQ(kStackTraceMax, trace->size);
  for (uptr i = 0; i < kStackTraceMax - 1; i++) {
    EXPECT_EQ(101 + kStackTraceMax + i, trace->trace[i]);
  }
  EXPECT_EQ(42U, trace->trace[kStackTraceMax - 1]);
}

TEST(StackTrace, BasicVarSize) {
  VarSizeStackTrace trace;
  TestStackTrace(&trace);
}

TEST(StackTrace, BasicBuffered) {
  BufferedStackTrace trace;
  TestStackTrace(&trace);
}

TEST(StackTrace, TrimVarSize) {
  VarSizeStackTrace trace;
  TestTrim(&trace);
}

TEST(StackTrace, TrimBuffered) {
  BufferedStackTrace trace;
  TestTrim(&trace);
}

}  // namespace __tsan
