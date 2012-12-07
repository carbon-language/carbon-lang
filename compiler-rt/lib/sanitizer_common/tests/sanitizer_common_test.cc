//===-- sanitizer_common_test.cc ------------------------------------------===//
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
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

namespace __sanitizer {

static bool IsSorted(const uptr *array, uptr n) {
  for (uptr i = 1; i < n; i++) {
    if (array[i] < array[i - 1]) return false;
  }
  return true;
}

TEST(SanitizerCommon, SortTest) {
  uptr array[100];
  uptr n = 100;
  // Already sorted.
  for (uptr i = 0; i < n; i++) {
    array[i] = i;
  }
  SortArray(array, n);
  EXPECT_TRUE(IsSorted(array, n));
  // Reverse order.
  for (uptr i = 0; i < n; i++) {
    array[i] = n - 1 - i;
  }
  SortArray(array, n);
  EXPECT_TRUE(IsSorted(array, n));
  // Mixed order.
  for (uptr i = 0; i < n; i++) {
    array[i] = (i % 2 == 0) ? i : n - 1 - i;
  }
  SortArray(array, n);
  EXPECT_TRUE(IsSorted(array, n));
  // All equal.
  for (uptr i = 0; i < n; i++) {
    array[i] = 42;
  }
  SortArray(array, n);
  EXPECT_TRUE(IsSorted(array, n));
  // All but one sorted.
  for (uptr i = 0; i < n - 1; i++) {
    array[i] = i;
  }
  array[n - 1] = 42;
  SortArray(array, n);
  EXPECT_TRUE(IsSorted(array, n));
  // Minimal case - sort three elements.
  array[0] = 1;
  array[1] = 0;
  SortArray(array, 2);
  EXPECT_TRUE(IsSorted(array, 2));
}

TEST(SanitizerCommon, MmapAlignedOrDie) {
  uptr PageSize = GetPageSizeCached();
  for (uptr size = 1; size <= 32; size *= 2) {
    for (uptr alignment = 1; alignment <= 32; alignment *= 2) {
      for (int iter = 0; iter < 100; iter++) {
        uptr res = (uptr)MmapAlignedOrDie(
            size * PageSize, alignment * PageSize, "MmapAlignedOrDieTest");
        EXPECT_EQ(0U, res % (alignment * PageSize));
        memset((void*)res, 1, size * PageSize);
        UnmapOrDie((void*)res, size * PageSize);
      }
    }
  }
}

#ifdef __linux__
TEST(SanitizerCommon, SanitizerSetThreadName) {
  const char *names[] = {
    "0123456789012",
    "01234567890123",
    "012345678901234",  // Larger names will be truncated on linux.
  };

  for (size_t i = 0; i < ARRAY_SIZE(names); i++) {
    EXPECT_TRUE(SanitizerSetThreadName(names[i]));
    char buff[100];
    EXPECT_TRUE(SanitizerGetThreadName(buff, sizeof(buff) - 1));
    Printf("buff: %s\n", buff);
    EXPECT_EQ(0, internal_strcmp(buff, names[i]));
  }
}
#endif

}  // namespace sanitizer
