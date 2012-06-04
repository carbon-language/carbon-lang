//===-- tsan_allocator_test.c----------------------------------------------===//
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
#include "tsan_allocator.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace __tsan {

TEST(Allocator, Basic) {
  char *p = (char*)Alloc(10);
  EXPECT_NE(p, (char*)0);
  char *p2 = (char*)Alloc(20);
  EXPECT_NE(p2, (char*)0);
  EXPECT_NE(p2, p);
  for (int i = 0; i < 10; i++) {
    p[i] = 42;
    EXPECT_EQ(p, AllocBlock(p + i));
  }
  for (int i = 0; i < 20; i++) {
    ((char*)p2)[i] = 42;
    EXPECT_EQ(p2, AllocBlock(p2 + i));
  }
  Free(p);
  Free(p2);
}

TEST(Allocator, Stress) {
  const int kCount = 1000;
  char *ptrs[kCount];
  unsigned rnd = 42;
  for (int i = 0; i < kCount; i++) {
    uptr sz = rand_r(&rnd) % 1000;
    char *p = (char*)Alloc(sz);
    EXPECT_NE(p, (char*)0);
    for (uptr j = 0; j < sz; j++) {
      p[j] = 42;
      EXPECT_EQ(p, AllocBlock(p + j));
    }
    ptrs[i] = p;
  }
  for (int i = 0; i < kCount; i++) {
    Free(ptrs[i]);
  }
}

}  // namespace __tsan
