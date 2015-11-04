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
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_platform.h"

#include "sanitizer_pthread_wrappers.h"

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
        internal_memset((void*)res, 1, size * PageSize);
        UnmapOrDie((void*)res, size * PageSize);
      }
    }
  }
}

#if SANITIZER_LINUX
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
    EXPECT_EQ(0, internal_strcmp(buff, names[i]));
  }
}
#endif

TEST(SanitizerCommon, InternalMmapVector) {
  InternalMmapVector<uptr> vector(1);
  for (uptr i = 0; i < 100; i++) {
    EXPECT_EQ(i, vector.size());
    vector.push_back(i);
  }
  for (uptr i = 0; i < 100; i++) {
    EXPECT_EQ(i, vector[i]);
  }
  for (int i = 99; i >= 0; i--) {
    EXPECT_EQ((uptr)i, vector.back());
    vector.pop_back();
    EXPECT_EQ((uptr)i, vector.size());
  }
  InternalMmapVector<uptr> empty_vector(0);
  CHECK_GT(empty_vector.capacity(), 0U);
  CHECK_EQ(0U, empty_vector.size());
}

void TestThreadInfo(bool main) {
  uptr stk_addr = 0;
  uptr stk_size = 0;
  uptr tls_addr = 0;
  uptr tls_size = 0;
  GetThreadStackAndTls(main, &stk_addr, &stk_size, &tls_addr, &tls_size);

  int stack_var;
  EXPECT_NE(stk_addr, (uptr)0);
  EXPECT_NE(stk_size, (uptr)0);
  EXPECT_GT((uptr)&stack_var, stk_addr);
  EXPECT_LT((uptr)&stack_var, stk_addr + stk_size);

#if SANITIZER_LINUX && defined(__x86_64__)
  static __thread int thread_var;
  EXPECT_NE(tls_addr, (uptr)0);
  EXPECT_NE(tls_size, (uptr)0);
  EXPECT_GT((uptr)&thread_var, tls_addr);
  EXPECT_LT((uptr)&thread_var, tls_addr + tls_size);

  // Ensure that tls and stack do not intersect.
  uptr tls_end = tls_addr + tls_size;
  EXPECT_TRUE(tls_addr < stk_addr || tls_addr >= stk_addr + stk_size);
  EXPECT_TRUE(tls_end  < stk_addr || tls_end  >=  stk_addr + stk_size);
  EXPECT_TRUE((tls_addr < stk_addr) == (tls_end  < stk_addr));
#endif
}

static void *WorkerThread(void *arg) {
  TestThreadInfo(false);
  return 0;
}

TEST(SanitizerCommon, ThreadStackTlsMain) {
  InitTlsSize();
  TestThreadInfo(true);
}

TEST(SanitizerCommon, ThreadStackTlsWorker) {
  InitTlsSize();
  pthread_t t;
  PTHREAD_CREATE(&t, 0, WorkerThread, 0);
  PTHREAD_JOIN(t, 0);
}

bool UptrLess(uptr a, uptr b) {
  return a < b;
}

TEST(SanitizerCommon, InternalBinarySearch) {
  static const uptr kSize = 5;
  uptr arr[kSize];
  for (uptr i = 0; i < kSize; i++) arr[i] = i * i;

  for (uptr i = 0; i < kSize; i++)
    ASSERT_EQ(InternalBinarySearch(arr, 0, kSize, i * i, UptrLess), i);

  ASSERT_EQ(InternalBinarySearch(arr, 0, kSize, 7, UptrLess), kSize + 1);
}

#if SANITIZER_LINUX && !SANITIZER_ANDROID
TEST(SanitizerCommon, FindPathToBinary) {
  char *true_path = FindPathToBinary("true");
  EXPECT_NE((char*)0, internal_strstr(true_path, "/bin/true"));
  InternalFree(true_path);
  EXPECT_EQ(0, FindPathToBinary("unexisting_binary.ergjeorj"));
}
#elif SANITIZER_WINDOWS
TEST(SanitizerCommon, FindPathToBinary) {
  // ntdll.dll should be on PATH in all supported test environments on all
  // supported Windows versions.
  char *ntdll_path = FindPathToBinary("ntdll.dll");
  EXPECT_NE((char*)0, internal_strstr(ntdll_path, "ntdll.dll"));
  InternalFree(ntdll_path);
  EXPECT_EQ(0, FindPathToBinary("unexisting_binary.ergjeorj"));
}
#endif

TEST(SanitizerCommon, StripPathPrefix) {
  EXPECT_EQ(0, StripPathPrefix(0, "prefix"));
  EXPECT_STREQ("foo", StripPathPrefix("foo", 0));
  EXPECT_STREQ("dir/file.cc",
               StripPathPrefix("/usr/lib/dir/file.cc", "/usr/lib/"));
  EXPECT_STREQ("/file.cc", StripPathPrefix("/usr/myroot/file.cc", "/myroot"));
  EXPECT_STREQ("file.h", StripPathPrefix("/usr/lib/./file.h", "/usr/lib/"));
}

TEST(SanitizerCommon, InternalScopedString) {
  InternalScopedString str(10);
  EXPECT_EQ(0U, str.length());
  EXPECT_STREQ("", str.data());

  str.append("foo");
  EXPECT_EQ(3U, str.length());
  EXPECT_STREQ("foo", str.data());

  int x = 1234;
  str.append("%d", x);
  EXPECT_EQ(7U, str.length());
  EXPECT_STREQ("foo1234", str.data());

  str.append("%d", x);
  EXPECT_EQ(9U, str.length());
  EXPECT_STREQ("foo123412", str.data());

  str.clear();
  EXPECT_EQ(0U, str.length());
  EXPECT_STREQ("", str.data());

  str.append("0123456789");
  EXPECT_EQ(9U, str.length());
  EXPECT_STREQ("012345678", str.data());
}

}  // namespace __sanitizer
