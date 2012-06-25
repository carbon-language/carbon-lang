//===-- sanitizer_allocator_test.cc ---------------------------------------===//
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
#include "sanitizer_common/sanitizer_common.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace __tsan {

TEST(Allocator, Basic) {
  char *p = (char*)InternalAlloc(10);
  EXPECT_NE(p, (char*)0);
  char *p2 = (char*)InternalAlloc(20);
  EXPECT_NE(p2, (char*)0);
  EXPECT_NE(p2, p);
  for (int i = 0; i < 10; i++) {
    p[i] = 42;
    EXPECT_EQ(p, InternalAllocBlock(p + i));
  }
  for (int i = 0; i < 20; i++) {
    ((char*)p2)[i] = 42;
    EXPECT_EQ(p2, InternalAllocBlock(p2 + i));
  }
  InternalFree(p);
  InternalFree(p2);
}

TEST(Allocator, Stress) {
  const int kCount = 1000;
  char *ptrs[kCount];
  unsigned rnd = 42;
  for (int i = 0; i < kCount; i++) {
    uptr sz = rand_r(&rnd) % 1000;
    char *p = (char*)InternalAlloc(sz);
    EXPECT_NE(p, (char*)0);
    for (uptr j = 0; j < sz; j++) {
      p[j] = 42;
      EXPECT_EQ(p, InternalAllocBlock(p + j));
    }
    ptrs[i] = p;
  }
  for (int i = 0; i < kCount; i++) {
    InternalFree(ptrs[i]);
  }
}

}  // namespace __tsan
