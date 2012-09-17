//===-- tsan_test.cc ------------------------------------------------------===//
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
#include "tsan_interface.h"
#include "tsan_test_util.h"
#include "gtest/gtest.h"

static void foo() {}
static void bar() {}

TEST(ThreadSanitizer, FuncCall) {
  ScopedThread t1, t2;
  MemLoc l;
  t1.Write1(l);
  t2.Call(foo);
  t2.Call(bar);
  t2.Write1(l, true);
  t2.Return();
  t2.Return();
}

int main(int argc, char **argv) {
  TestMutexBeforeInit();  // Mutexes must be usable before __tsan_init();
  __tsan_init();
  __tsan_func_entry(__builtin_return_address(0));
  __tsan_func_entry((char*)&main + 1);

  testing::GTEST_FLAG(death_test_style) = "threadsafe";
  testing::InitGoogleTest(&argc, argv);
  int res = RUN_ALL_TESTS();

  __tsan_func_exit();
  __tsan_func_exit();
  return res;
}
