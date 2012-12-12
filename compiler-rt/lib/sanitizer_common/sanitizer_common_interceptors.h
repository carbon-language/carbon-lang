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

typedef uptr size_t;
typedef sptr ssize_t;
typedef u64  off_t;
typedef u64  off64_t;

INTERCEPTOR(ssize_t, read, int fd, void *ptr, size_t count) {
  COMMON_INTERCEPTOR_ENTER(read, fd, ptr, count);
  ssize_t res = REAL(read)(fd, ptr, count);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  return res;
}

INTERCEPTOR(ssize_t, pread, int fd, void *ptr, size_t count, off_t offset) {
  COMMON_INTERCEPTOR_ENTER(pread, fd, ptr, count, offset);
  ssize_t res = REAL(pread)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  return res;
}

INTERCEPTOR(ssize_t, pread64, int fd, void *ptr, size_t count, off64_t offset) {
  COMMON_INTERCEPTOR_ENTER(pread64, fd, ptr, count, offset);
  ssize_t res = REAL(pread64)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ptr, res);
  return res;
}

#define SANITIZER_COMMON_INTERCEPTORS_INIT \
  CHECK(INTERCEPT_FUNCTION(read));         \
  CHECK(INTERCEPT_FUNCTION(pread));        \
  CHECK(INTERCEPT_FUNCTION(pread64))       \
  ;

#endif  // SANITIZER_COMMON_INTERCEPTORS_H
