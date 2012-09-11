//===-- sanitizer_stackdepot_test.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

namespace __sanitizer {

TEST(SanitizerCommon, StackDepotBasic) {
  uptr s1[] = {1, 2, 3, 4, 5};
  u32 i1 = StackDepotPut(s1, ARRAY_SIZE(s1));
  uptr sz1 = 0;
  const uptr *sp1 = StackDepotGet(i1, &sz1);
  EXPECT_NE(sp1, (uptr*)0);
  EXPECT_EQ(sz1, ARRAY_SIZE(s1));
  EXPECT_EQ(internal_memcmp(sp1, s1, sizeof(s1)), 0);
}

TEST(SanitizerCommon, StackDepotAbsent) {
  uptr sz1 = 0;
  const uptr *sp1 = StackDepotGet(-10, &sz1);
  EXPECT_EQ(sp1, (uptr*)0);
}

TEST(SanitizerCommon, StackDepotEmptyStack) {
  u32 i1 = StackDepotPut(0, 0);
  uptr sz1 = 0;
  const uptr *sp1 = StackDepotGet(i1, &sz1);
  EXPECT_EQ(sp1, (uptr*)0);
}

TEST(SanitizerCommon, StackDepotZeroId) {
  uptr sz1 = 0;
  const uptr *sp1 = StackDepotGet(0, &sz1);
  EXPECT_EQ(sp1, (uptr*)0);
}

TEST(SanitizerCommon, StackDepotSame) {
  uptr s1[] = {1, 2, 3, 4, 6};
  u32 i1 = StackDepotPut(s1, ARRAY_SIZE(s1));
  u32 i2 = StackDepotPut(s1, ARRAY_SIZE(s1));
  EXPECT_EQ(i1, i2);
  uptr sz1 = 0;
  const uptr *sp1 = StackDepotGet(i1, &sz1);
  EXPECT_NE(sp1, (uptr*)0);
  EXPECT_EQ(sz1, ARRAY_SIZE(s1));
  EXPECT_EQ(internal_memcmp(sp1, s1, sizeof(s1)), 0);
}

TEST(SanitizerCommon, StackDepotSeveral) {
  uptr s1[] = {1, 2, 3, 4, 7};
  u32 i1 = StackDepotPut(s1, ARRAY_SIZE(s1));
  uptr s2[] = {1, 2, 3, 4, 8, 9};
  u32 i2 = StackDepotPut(s2, ARRAY_SIZE(s2));
  EXPECT_NE(i1, i2);
}

}  // namespace __sanitizer
