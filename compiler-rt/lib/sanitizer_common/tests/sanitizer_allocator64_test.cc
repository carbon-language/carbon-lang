//===-- sanitizer_allocator64_test.cc -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Tests for sanitizer_allocator64.h.
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_allocator64.h"
#include "gtest/gtest.h"

TEST(SanitizerCommon, DefaultSizeClassMap) {
  typedef DefaultSizeClassMap SCMap;

  for (uptr i = 0; i < SCMap::kNumClasses; i++) {
    // printf("% 3ld: % 5ld (%4lx);   ", i, SCMap::Size(i), SCMap::Size(i));
    printf("c%ld => %ld  ", i, SCMap::Size(i));
    if ((i % 8) == 7)
      printf("\n");
  }
  printf("\n");

  for (uptr c = 0; c < SCMap::kNumClasses; c++) {
    uptr s = SCMap::Size(c);
    CHECK_EQ(SCMap::Class(s), c);
    if (c != SCMap::kNumClasses - 1)
      CHECK_EQ(SCMap::Class(s + 1), c + 1);
    CHECK_EQ(SCMap::Class(s - 1), c);
    if (c)
      CHECK_GT(SCMap::Size(c), SCMap::Size(c-1));
  }
  CHECK_EQ(SCMap::Class(SCMap::kMaxSize + 1), 0);

  for (uptr s = 1; s <= SCMap::kMaxSize; s++) {
    uptr c = SCMap::Class(s);
    CHECK_LT(c, SCMap::kNumClasses);
    CHECK_GE(SCMap::Size(c), s);
    if (c > 0)
      CHECK_LT(SCMap::Size(c-1), s);
  }
}
