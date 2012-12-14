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
//
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
  return res;
}
#endif

#if SANITIZER_INTERCEPT_PREAD
INTERCEPTOR(SSIZE_T, pread, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  COMMON_INTERCEPTOR_ENTER(pread, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  return res;
}
#endif

#if SANITIZER_INTERCEPT_PREAD64
INTERCEPTOR(SSIZE_T, pread64, int fd, void *ptr, SIZE_T count, OFF64_T offset) {
  COMMON_INTERCEPTOR_ENTER(pread64, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread64)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  return res;
}
#endif

#if SANITIZER_INTERCEPT_READ
# define INIT_READ INTERCEPT_FUNCTION(read)
#else
# define INIT_READ
#endif

#if SANITIZER_INTERCEPT_PREAD
# define INIT_PREAD INTERCEPT_FUNCTION(pread)
#else
# define INIT_PREAD
#endif

#if SANITIZER_INTERCEPT_PREAD64
# define INIT_PREAD64 INTERCEPT_FUNCTION(pread64)
#else
# define INIT_PREAD64
#endif

#define SANITIZER_COMMON_INTERCEPTORS_INIT \
  INIT_READ;                               \
  INIT_PREAD;                              \
  INIT_PREAD64;                            \

#endif  // SANITIZER_COMMON_INTERCEPTORS_H
