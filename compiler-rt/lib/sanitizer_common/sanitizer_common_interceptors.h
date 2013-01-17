//===-- sanitizer_common_interceptors.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common function interceptors for tools like AddressSanitizer,
// ThreadSanitizer, MemorySanitizer, etc.
//
// This file should be included into the tool's interceptor file,
// which has to define it's own macros:
//   COMMON_INTERCEPTOR_ENTER
//   COMMON_INTERCEPTOR_READ_RANGE
//   COMMON_INTERCEPTOR_WRITE_RANGE
//   COMMON_INTERCEPTOR_FD_ACQUIRE
//   COMMON_INTERCEPTOR_FD_RELEASE
//   COMMON_INTERCEPTOR_SET_THREAD_NAME
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_COMMON_INTERCEPTORS_H
#define SANITIZER_COMMON_INTERCEPTORS_H

#include "interception/interception.h"
#include "sanitizer_platform_interceptors.h"

#if SANITIZER_INTERCEPT_READ
INTERCEPTOR(SSIZE_T, read, int fd, void *ptr, SIZE_T count) {
  COMMON_INTERCEPTOR_ENTER(read, fd, ptr, count);
  SSIZE_T res = REAL(read)(fd, ptr, count);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(fd);
  return res;
}
# define INIT_READ INTERCEPT_FUNCTION(read)
#else
# define INIT_READ
#endif

#if SANITIZER_INTERCEPT_PREAD
INTERCEPTOR(SSIZE_T, pread, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  COMMON_INTERCEPTOR_ENTER(pread, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(fd);
  return res;
}
# define INIT_PREAD INTERCEPT_FUNCTION(pread)
#else
# define INIT_PREAD
#endif

#if SANITIZER_INTERCEPT_PREAD64
INTERCEPTOR(SSIZE_T, pread64, int fd, void *ptr, SIZE_T count, OFF64_T offset) {
  COMMON_INTERCEPTOR_ENTER(pread64, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread64)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(fd);
  return res;
}
# define INIT_PREAD64 INTERCEPT_FUNCTION(pread64)
#else
# define INIT_PREAD64
#endif

#if SANITIZER_INTERCEPT_PRCTL
INTERCEPTOR(int, prctl, int option,
            unsigned long arg2, unsigned long arg3,  // NOLINT
            unsigned long arg4, unsigned long arg5) {  // NOLINT
  COMMON_INTERCEPTOR_ENTER(prctl, option, arg2, arg3, arg4, arg5);
  static const int PR_SET_NAME = 15;
  int res = REAL(prctl(option, arg2, arg3, arg4, arg5));
  if (option == PR_SET_NAME) {
    char buff[16];
    internal_strncpy(buff, (char*)arg2, 15);
    buff[15] = 0;
    COMMON_INTERCEPTOR_SET_THREAD_NAME(buff);
  }
  return res;
}
# define INIT_PRCTL INTERCEPT_FUNCTION(prctl)
#else
# define INIT_PRCTL
#endif  // SANITIZER_INTERCEPT_PRCTL

#define SANITIZER_COMMON_INTERCEPTORS_INIT \
  INIT_READ;                               \
  INIT_PREAD;                              \
  INIT_PREAD64;                            \
  INIT_PRCTL;                              \

#endif  // SANITIZER_COMMON_INTERCEPTORS_H
