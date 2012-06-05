//===-- tsan_platform_test.cc ---------------------------------------------===//
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
#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_platform.h"
#include "gtest/gtest.h"

using namespace __sanitizer;  // NOLINT

namespace __tsan {

static void TestThreadInfo(bool main) {
  ScopedInRtl in_rtl;
  uptr stk_addr = 0;
  uptr stk_size = 0;
  uptr tls_addr = 0;
  uptr tls_size = 0;
  GetThreadStackAndTls(main, &stk_addr, &stk_size, &tls_addr, &tls_size);
  // Printf("stk=%lx-%lx(%lu)\n", stk_addr, stk_addr + stk_size, stk_size);
  // Printf("tls=%lx-%lx(%lu)\n", tls_addr, tls_addr + tls_size, tls_size);

  int stack_var;
  EXPECT_NE(stk_addr, (uptr)0);
  EXPECT_NE(stk_size, (uptr)0);
  EXPECT_GT((uptr)&stack_var, stk_addr);
  EXPECT_LT((uptr)&stack_var, stk_addr + stk_size);

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
}

static void *WorkerThread(void *arg) {
  TestThreadInfo(false);
  return 0;
}

TEST(Platform, ThreadInfoMain) {
  TestThreadInfo(true);
}

TEST(Platform, ThreadInfoWorker) {
  pthread_t t;
  pthread_create(&t, 0, WorkerThread, 0);
  pthread_join(t, 0);
}

TEST(Platform, FileOps) {
  const char *str1 = "qwerty";
  uptr len1 = internal_strlen(str1);
  const char *str2 = "zxcv";
  uptr len2 = internal_strlen(str2);

  fd_t fd = internal_open("./tsan_test.tmp", true);
  EXPECT_NE(fd, kInvalidFd);
  EXPECT_EQ(len1, internal_write(fd, str1, len1));
  EXPECT_EQ(len2, internal_write(fd, str2, len2));
  internal_close(fd);

  fd = internal_open("./tsan_test.tmp", false);
  EXPECT_NE(fd, kInvalidFd);
  EXPECT_EQ(len1 + len2, internal_filesize(fd));
  char buf[64] = {};
  EXPECT_EQ(len1, internal_read(fd, buf, len1));
  EXPECT_EQ(0, internal_memcmp(buf, str1, len1));
  EXPECT_EQ((char)0, buf[len1 + 1]);
  internal_memset(buf, 0, len1);
  EXPECT_EQ(len2, internal_read(fd, buf, len2));
  EXPECT_EQ(0, internal_memcmp(buf, str2, len2));
  internal_close(fd);
}

}  // namespace __tsan
