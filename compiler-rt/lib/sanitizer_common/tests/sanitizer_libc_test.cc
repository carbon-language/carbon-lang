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
#include "sanitizer_common/sanitizer_platform.h"
#include "gtest/gtest.h"

#if SANITIZER_LINUX || SANITIZER_MAC
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

struct stat_and_more {
  struct stat st;
  unsigned char z;
};

static void temp_file_name(char *buf, size_t bufsize, const char *prefix) {
  const char *tmpdir = "/tmp";
#if SANITIZER_ANDROID
  // I don't know a way to query temp directory location on Android without
  // going through Java interfaces. The code below is not ideal, but should
  // work. May require "adb root", but it is needed for almost any use of ASan
  // on Android already.
  tmpdir = GetEnv("EXTERNAL_STORAGE");
#endif
  u32 uid = GetUid();
  internal_snprintf(buf, bufsize, "%s/%s%d", tmpdir, prefix, uid);
}

// FIXME: File manipulations are not yet supported on Windows
#if !defined(_WIN32)
TEST(SanitizerCommon, FileOps) {
  const char *str1 = "qwerty";
  uptr len1 = internal_strlen(str1);
  const char *str2 = "zxcv";
  uptr len2 = internal_strlen(str2);

  char tmpfile[128];
  temp_file_name(tmpfile, sizeof(tmpfile), "sanitizer_common.fileops.tmp.");
  uptr openrv = OpenFile(tmpfile, true);
  EXPECT_FALSE(internal_iserror(openrv));
  fd_t fd = openrv;
  EXPECT_EQ(len1, internal_write(fd, str1, len1));
  EXPECT_EQ(len2, internal_write(fd, str2, len2));
  internal_close(fd);

  openrv = OpenFile(tmpfile, false);
  EXPECT_FALSE(internal_iserror(openrv));
  fd = openrv;
  uptr fsize = internal_filesize(fd);
  EXPECT_EQ(len1 + len2, fsize);

#if SANITIZER_TEST_HAS_STAT_H
  struct stat st1, st2, st3;
  EXPECT_EQ(0u, internal_stat(tmpfile, &st1));
  EXPECT_EQ(0u, internal_lstat(tmpfile, &st2));
  EXPECT_EQ(0u, internal_fstat(fd, &st3));
  EXPECT_EQ(fsize, (uptr)st3.st_size);

  // Verify that internal_fstat does not write beyond the end of the supplied
  // buffer.
  struct stat_and_more sam;
  memset(&sam, 0xAB, sizeof(sam));
  EXPECT_EQ(0u, internal_fstat(fd, &sam.st));
  EXPECT_EQ(0xAB, sam.z);
  EXPECT_NE(0xAB, sam.st.st_size);
  EXPECT_NE(0, sam.st.st_size);
#endif

  char buf[64] = {};
  EXPECT_EQ(len1, internal_read(fd, buf, len1));
  EXPECT_EQ(0, internal_memcmp(buf, str1, len1));
  EXPECT_EQ((char)0, buf[len1 + 1]);
  internal_memset(buf, 0, len1);
  EXPECT_EQ(len2, internal_read(fd, buf, len2));
  EXPECT_EQ(0, internal_memcmp(buf, str2, len2));
  internal_close(fd);
  internal_unlink(tmpfile);
}
#endif

TEST(SanitizerCommon, InternalStrFunctions) {
  const char *haystack = "haystack";
  EXPECT_EQ(haystack + 2, internal_strchr(haystack, 'y'));
  EXPECT_EQ(haystack + 2, internal_strchrnul(haystack, 'y'));
  EXPECT_EQ(0, internal_strchr(haystack, 'z'));
  EXPECT_EQ(haystack + 8, internal_strchrnul(haystack, 'z'));
}

// FIXME: File manipulations are not yet supported on Windows
#if !defined(_WIN32) && !SANITIZER_MAC
TEST(SanitizerCommon, InternalMmapWithOffset) {
  char tmpfile[128];
  temp_file_name(tmpfile, sizeof(tmpfile),
                 "sanitizer_common.internalmmapwithoffset.tmp.");
  uptr res = OpenFile(tmpfile, true);
  ASSERT_FALSE(internal_iserror(res));
  fd_t fd = res;

  uptr page_size = GetPageSizeCached();
  res = internal_ftruncate(fd, page_size * 2);
  ASSERT_FALSE(internal_iserror(res));

  res = internal_lseek(fd, page_size, SEEK_SET);
  ASSERT_FALSE(internal_iserror(res));

  res = internal_write(fd, "AB", 2);
  ASSERT_FALSE(internal_iserror(res));

  char *p = (char *)MapWritableFileToMemory(nullptr, page_size, fd, page_size);
  ASSERT_NE(nullptr, p);

  ASSERT_EQ('A', p[0]);
  ASSERT_EQ('B', p[1]);

  internal_close(fd);
  internal_munmap(p, page_size);
  internal_unlink(tmpfile);
}
#endif
