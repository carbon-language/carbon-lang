//===-- sanitizer_libc_test.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Tests for sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

#if defined(__linux__) || defined(__APPLE__)
# define SANITIZER_TEST_HAS_STAT_H 1
# include <sys/stat.h>
#else
# define SANITIZER_TEST_HAS_STAT_H 0
#endif

// A regression test for internal_memmove() implementation.
TEST(SanitizerCommon, InternalMemmoveRegression) {
  char src[] = "Hello World";
  char *dest = src + 6;
  __sanitizer::internal_memmove(dest, src, 5);
  EXPECT_EQ(dest[0], src[0]);
  EXPECT_EQ(dest[4], src[4]);
}

TEST(SanitizerCommon, mem_is_zero) {
  size_t size = 128;
  char *x = new char[size];
  memset(x, 0, size);
  for (size_t pos = 0; pos < size; pos++) {
    x[pos] = 1;
    for (size_t beg = 0; beg < size; beg++) {
      for (size_t end = beg; end < size; end++) {
        // fprintf(stderr, "pos %zd beg %zd end %zd \n", pos, beg, end);
        if (beg <= pos && pos < end)
          EXPECT_FALSE(__sanitizer::mem_is_zero(x + beg, end - beg));
        else
          EXPECT_TRUE(__sanitizer::mem_is_zero(x + beg, end - beg));
      }
    }
    x[pos] = 0;
  }
  delete [] x;
}

TEST(SanitizerCommon, FileOps) {
  const char *str1 = "qwerty";
  uptr len1 = internal_strlen(str1);
  const char *str2 = "zxcv";
  uptr len2 = internal_strlen(str2);

  u32 uid = GetUid();
  char temp_filename[128];
#ifdef __ANDROID__
  // I don't know a way to query temp directory location on Android without
  // going through Java interfaces. The code below is not ideal, but should
  // work. May require "adb root", but it is needed for almost any use of ASan
  // on Android already.
  internal_snprintf(temp_filename, sizeof(temp_filename),
                    "%s/sanitizer_common.tmp.%d",
                    GetEnv("EXTERNAL_STORAGE"), uid);
#else
  internal_snprintf(temp_filename, sizeof(temp_filename),
                    "/tmp/sanitizer_common.tmp.%d", uid);
#endif
  fd_t fd = OpenFile(temp_filename, true);
  EXPECT_NE(fd, kInvalidFd);
  EXPECT_EQ(len1, internal_write(fd, str1, len1));
  EXPECT_EQ(len2, internal_write(fd, str2, len2));
  internal_close(fd);

  fd = OpenFile(temp_filename, false);
  EXPECT_NE(fd, kInvalidFd);
  uptr fsize = internal_filesize(fd);
  EXPECT_EQ(len1 + len2, fsize);

#if SANITIZER_TEST_HAS_STAT_H
  struct stat st1, st2, st3;
  EXPECT_EQ(0, internal_stat(temp_filename, &st1));
  EXPECT_EQ(0, internal_lstat(temp_filename, &st2));
  EXPECT_EQ(0, internal_fstat(fd, &st3));
  EXPECT_EQ(fsize, (uptr)st3.st_size);
#endif

  char buf[64] = {};
  EXPECT_EQ(len1, internal_read(fd, buf, len1));
  EXPECT_EQ(0, internal_memcmp(buf, str1, len1));
  EXPECT_EQ((char)0, buf[len1 + 1]);
  internal_memset(buf, 0, len1);
  EXPECT_EQ(len2, internal_read(fd, buf, len2));
  EXPECT_EQ(0, internal_memcmp(buf, str2, len2));
  internal_close(fd);
}
