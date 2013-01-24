//===-- sanitizer_common_interceptors.inc -----------------------*- C++ -*-===//
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
#include "interception/interception.h"
#include "sanitizer_platform_interceptors.h"

#include <stdarg.h>

#if SANITIZER_INTERCEPT_READ
INTERCEPTOR(SSIZE_T, read, int fd, void *ptr, SIZE_T count) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, read, fd, ptr, count);
  SSIZE_T res = REAL(read)(fd, ptr, count);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
# define INIT_READ INTERCEPT_FUNCTION(read)
#else
# define INIT_READ
#endif

#if SANITIZER_INTERCEPT_PREAD
INTERCEPTOR(SSIZE_T, pread, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pread, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
# define INIT_PREAD INTERCEPT_FUNCTION(pread)
#else
# define INIT_PREAD
#endif

#if SANITIZER_INTERCEPT_PREAD64
INTERCEPTOR(SSIZE_T, pread64, int fd, void *ptr, SIZE_T count, OFF64_T offset) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pread64, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread64)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
# define INIT_PREAD64 INTERCEPT_FUNCTION(pread64)
#else
# define INIT_PREAD64
#endif

#if SANITIZER_INTERCEPT_WRITE
INTERCEPTOR(SSIZE_T, write, int fd, void *ptr, SIZE_T count) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, write, fd, ptr, count);
  if (fd >= 0)
    COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(write)(fd, ptr, count);
  if (res > 0)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
# define INIT_WRITE INTERCEPT_FUNCTION(write)
#else
# define INIT_WRITE
#endif

#if SANITIZER_INTERCEPT_PWRITE
INTERCEPTOR(SSIZE_T, pwrite, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwrite, fd, ptr, count, offset);
  if (fd >= 0)
    COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwrite)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
# define INIT_PWRITE INTERCEPT_FUNCTION(pwrite)
#else
# define INIT_PWRITE
#endif

#if SANITIZER_INTERCEPT_PWRITE64
INTERCEPTOR(SSIZE_T, pwrite64, int fd, void *ptr, OFF64_T count,
            OFF64_T offset) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwrite64, fd, ptr, count, offset);
  if (fd >= 0)
    COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwrite64)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
# define INIT_PWRITE64 INTERCEPT_FUNCTION(pwrite64)
#else
# define INIT_PWRITE64
#endif

#if SANITIZER_INTERCEPT_PRCTL
INTERCEPTOR(int, prctl, int option,
            unsigned long arg2, unsigned long arg3,  // NOLINT
            unsigned long arg4, unsigned long arg5) {  // NOLINT
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, prctl, option, arg2, arg3, arg4, arg5);
  static const int PR_SET_NAME = 15;
  int res = REAL(prctl(option, arg2, arg3, arg4, arg5));
  if (option == PR_SET_NAME) {
    char buff[16];
    internal_strncpy(buff, (char*)arg2, 15);
    buff[15] = 0;
    COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, buff);
  }
  return res;
}
# define INIT_PRCTL INTERCEPT_FUNCTION(prctl)
#else
# define INIT_PRCTL
#endif  // SANITIZER_INTERCEPT_PRCTL


#if SANITIZER_INTERCEPT_SCANF

#include "sanitizer_common_interceptors_scanf.inc"

INTERCEPTOR(int, vscanf, const char *format, va_list ap) {  // NOLINT
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, vscanf, format, ap);
  scanf_common(ctx, format, ap);
  int res = REAL(vscanf)(format, ap);  // NOLINT
  return res;
}

INTERCEPTOR(int, vsscanf, const char *str, const char *format,  // NOLINT
    va_list ap) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, vsscanf, str, format, ap);
  scanf_common(ctx, format, ap);
  int res = REAL(vsscanf)(str, format, ap);  // NOLINT
  // FIXME: read of str
  return res;
}

INTERCEPTOR(int, vfscanf, void *stream, const char *format,  // NOLINT
    va_list ap) {
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, vfscanf, stream, format, ap);
  scanf_common(ctx, format, ap);
  int res = REAL(vfscanf)(stream, format, ap);  // NOLINT
  return res;
}

INTERCEPTOR(int, scanf, const char *format, ...) {  // NOLINT
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, scanf, format);
  va_list ap;
  va_start(ap, format);
  int res = vscanf(format, ap);  // NOLINT
  va_end(ap);
  return res;
}

INTERCEPTOR(int, fscanf, void* stream, const char *format, ...) {  // NOLINT
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fscanf, stream, format);
  va_list ap;
  va_start(ap, format);
  int res = vfscanf(stream, format, ap);  // NOLINT
  va_end(ap);
  return res;
}

INTERCEPTOR(int, sscanf, const char *str, const char *format, ...) {  // NOLINT
  void* ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sscanf, str, format);  // NOLINT
  va_list ap;
  va_start(ap, format);
  int res = vsscanf(str, format, ap);  // NOLINT
  va_end(ap);
  return res;
}

#define INIT_SCANF \
  INTERCEPT_FUNCTION(scanf);                    \
  INTERCEPT_FUNCTION(sscanf);  /* NOLINT */     \
  INTERCEPT_FUNCTION(fscanf);                   \
  INTERCEPT_FUNCTION(vscanf);                   \
  INTERCEPT_FUNCTION(vsscanf);                  \
  INTERCEPT_FUNCTION(vfscanf)

#else
#define INIT_SCANF
#endif

#define SANITIZER_COMMON_INTERCEPTORS_INIT \
  INIT_READ;                               \
  INIT_PREAD;                              \
  INIT_PREAD64;                            \
  INIT_PRCTL;                              \
  INIT_WRITE;                              \
  INIT_PWRITE;                             \
  INIT_PWRITE64;                           \
  INIT_SCANF;
