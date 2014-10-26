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
  StackTrace stack = StackDepotGet(i1);
  EXPECT_NE(stack.trace, (uptr*)0);
  EXPECT_EQ(ARRAY_SIZE(s1), stack.size);
  EXPECT_EQ(0, internal_memcmp(stack.trace, s1, sizeof(s1)));
}

TEST(SanitizerCommon, StackDepotAbsent) {
  StackTrace stack = StackDepotGet((1 << 30) - 1);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST(SanitizerCommon, StackDepotEmptyStack) {
  u32 i1 = StackDepotPut(0, 0);
  StackTrace stack = StackDepotGet(i1);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST(SanitizerCommon, StackDepotZeroId) {
  StackTrace stack = StackDepotGet(0);
  EXPECT_EQ((uptr*)0, stack.trace);
}

TEST(SanitizerCommon, StackDepotSame) {
  uptr s1[] = {1, 2, 3, 4, 6};
  u32 i1 = StackDepotPut(s1, ARRAY_SIZE(s1));
  u32 i2 = StackDepotPut(s1, ARRAY_SIZE(s1));
  EXPECT_EQ(i1, i2);
  StackTrace stack = StackDepotGet(i1);
  EXPECT_NE(stack.trace, (uptr*)0);
  EXPECT_EQ(ARRAY_SIZE(s1), stack.size);
  EXPECT_EQ(0, internal_memcmp(stack.trace, s1, sizeof(s1)));
}

TEST(SanitizerCommon, StackDepotSeveral) {
  uptr s1[] = {1, 2, 3, 4, 7};
  u32 i1 = StackDepotPut(s1, ARRAY_SIZE(s1));
  uptr s2[] = {1, 2, 3, 4, 8, 9};
  u32 i2 = StackDepotPut(s2, ARRAY_SIZE(s2));
  EXPECT_NE(i1, i2);
}

TEST(SanitizerCommon, StackDepotReverseMap) {
  uptr s1[] = {1, 2, 3, 4, 5};
  uptr s2[] = {7, 1, 3, 0};
  uptr s3[] = {10, 2, 5, 3};
  uptr s4[] = {1, 3, 2, 5};
  u32 ids[4] = {0};
  ids[0] = StackDepotPut(s1, ARRAY_SIZE(s1));
  ids[1] = StackDepotPut(s2, ARRAY_SIZE(s2));
  ids[2] = StackDepotPut(s3, ARRAY_SIZE(s3));
  ids[3] = StackDepotPut(s4, ARRAY_SIZE(s4));

  StackDepotReverseMap map;

  for (uptr i = 0; i < 4; i++) {
    uptr sz_map;
    const uptr *sp_map;
    StackTrace stack = StackDepotGet(ids[i]);
    sp_map = map.Get(ids[i], &sz_map);
    EXPECT_EQ(stack.size, sz_map);
    EXPECT_EQ(stack.trace, sp_map);
  }
}

}  // namespace __sanitizer
