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

#if SANITIZER_WINDOWS
#define va_copy(dst, src) ((dst) = (src))
#endif // _WIN32

#if SANITIZER_INTERCEPT_STRCASECMP
static inline int CharCaseCmp(unsigned char c1, unsigned char c2) {
  int c1_low = ToLower(c1);
  int c2_low = ToLower(c2);
  return c1_low - c2_low;
}

INTERCEPTOR(int, strcasecmp, const char *s1, const char *s2) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strcasecmp, s1, s2);
  unsigned char c1 = 0, c2 = 0;
  uptr i;
  for (i = 0; ; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0')
      break;
  }
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s1, i + 1);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s2, i + 1);
  return CharCaseCmp(c1, c2);
}

INTERCEPTOR(int, strncasecmp, const char *s1, const char *s2, SIZE_T n) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strncasecmp, s1, s2, n);
  unsigned char c1 = 0, c2 = 0;
  uptr i;
  for (i = 0; i < n; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0')
      break;
  }
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s1, Min(i + 1, n));
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s2, Min(i + 1, n));
  return CharCaseCmp(c1, c2);
}

#define INIT_STRCASECMP INTERCEPT_FUNCTION(strcasecmp)
#define INIT_STRNCASECMP INTERCEPT_FUNCTION(strncasecmp)
#else
#define INIT_STRCASECMP
#define INIT_STRNCASECMP
#endif

#if SANITIZER_INTERCEPT_FREXP
INTERCEPTOR(double, frexp, double x, int *exp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, frexp, x, exp);
  double res = REAL(frexp)(x, exp);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, exp, sizeof(*exp));
  return res;
}

#define INIT_FREXP INTERCEPT_FUNCTION(frexp);
#else
#define INIT_FREXP
#endif // SANITIZER_INTERCEPT_FREXP

#if SANITIZER_INTERCEPT_FREXPF_FREXPL
INTERCEPTOR(float, frexpf, float x, int *exp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, frexpf, x, exp);
  float res = REAL(frexpf)(x, exp);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, exp, sizeof(*exp));
  return res;
}

INTERCEPTOR(long double, frexpl, long double x, int *exp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, frexpl, x, exp);
  long double res = REAL(frexpl)(x, exp);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, exp, sizeof(*exp));
  return res;
}

#define INIT_FREXPF_FREXPL                       \
  INTERCEPT_FUNCTION(frexpf);                    \
  INTERCEPT_FUNCTION(frexpl)
#else
#define INIT_FREXPF_FREXPL
#endif // SANITIZER_INTERCEPT_FREXPF_FREXPL

#if SI_NOT_WINDOWS
static void write_iovec(void *ctx, struct __sanitizer_iovec *iovec,
                        SIZE_T iovlen, SIZE_T maxlen) {
  for (SIZE_T i = 0; i < iovlen && maxlen; ++i) {
    SSIZE_T sz = Min(iovec[i].iov_len, maxlen);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, iovec[i].iov_base, sz);
    maxlen -= sz;
  }
}

static void read_iovec(void *ctx, struct __sanitizer_iovec *iovec,
                       SIZE_T iovlen, SIZE_T maxlen) {
  COMMON_INTERCEPTOR_READ_RANGE(ctx, iovec, sizeof(*iovec) * iovlen);
  for (SIZE_T i = 0; i < iovlen && maxlen; ++i) {
    SSIZE_T sz = Min(iovec[i].iov_len, maxlen);
    COMMON_INTERCEPTOR_READ_RANGE(ctx, iovec[i].iov_base, sz);
    maxlen -= sz;
  }
}
#endif

#if SANITIZER_INTERCEPT_READ
INTERCEPTOR(SSIZE_T, read, int fd, void *ptr, SIZE_T count) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, read, fd, ptr, count);
  SSIZE_T res = REAL(read)(fd, ptr, count);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_READ INTERCEPT_FUNCTION(read)
#else
#define INIT_READ
#endif

#if SANITIZER_INTERCEPT_PREAD
INTERCEPTOR(SSIZE_T, pread, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pread, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREAD INTERCEPT_FUNCTION(pread)
#else
#define INIT_PREAD
#endif

#if SANITIZER_INTERCEPT_PREAD64
INTERCEPTOR(SSIZE_T, pread64, int fd, void *ptr, SIZE_T count, OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pread64, fd, ptr, count, offset);
  SSIZE_T res = REAL(pread64)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0)
    COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREAD64 INTERCEPT_FUNCTION(pread64)
#else
#define INIT_PREAD64
#endif

#if SANITIZER_INTERCEPT_READV
INTERCEPTOR_WITH_SUFFIX(SSIZE_T, readv, int fd, __sanitizer_iovec *iov,
                        int iovcnt) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readv, fd, iov, iovcnt);
  SSIZE_T res = REAL(readv)(fd, iov, iovcnt);
  if (res > 0) write_iovec(ctx, iov, iovcnt, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_READV INTERCEPT_FUNCTION(readv)
#else
#define INIT_READV
#endif

#if SANITIZER_INTERCEPT_PREADV
INTERCEPTOR(SSIZE_T, preadv, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, preadv, fd, iov, iovcnt, offset);
  SSIZE_T res = REAL(preadv)(fd, iov, iovcnt, offset);
  if (res > 0) write_iovec(ctx, iov, iovcnt, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREADV INTERCEPT_FUNCTION(preadv)
#else
#define INIT_PREADV
#endif

#if SANITIZER_INTERCEPT_PREADV64
INTERCEPTOR(SSIZE_T, preadv64, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, preadv64, fd, iov, iovcnt, offset);
  SSIZE_T res = REAL(preadv64)(fd, iov, iovcnt, offset);
  if (res > 0) write_iovec(ctx, iov, iovcnt, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREADV64 INTERCEPT_FUNCTION(preadv64)
#else
#define INIT_PREADV64
#endif

#if SANITIZER_INTERCEPT_WRITE
INTERCEPTOR(SSIZE_T, write, int fd, void *ptr, SIZE_T count) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, write, fd, ptr, count);
  if (fd >= 0)
    COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(write)(fd, ptr, count);
  // FIXME: this check should be _before_ the call to REAL(write), not after
  if (res > 0)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
#define INIT_WRITE INTERCEPT_FUNCTION(write)
#else
#define INIT_WRITE
#endif

#if SANITIZER_INTERCEPT_PWRITE
INTERCEPTOR(SSIZE_T, pwrite, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwrite, fd, ptr, count, offset);
  if (fd >= 0)
    COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwrite)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
#define INIT_PWRITE INTERCEPT_FUNCTION(pwrite)
#else
#define INIT_PWRITE
#endif

#if SANITIZER_INTERCEPT_PWRITE64
INTERCEPTOR(SSIZE_T, pwrite64, int fd, void *ptr, OFF64_T count,
            OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwrite64, fd, ptr, count, offset);
  if (fd >= 0)
    COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwrite64)(fd, ptr, count, offset);
  if (res > 0)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
#define INIT_PWRITE64 INTERCEPT_FUNCTION(pwrite64)
#else
#define INIT_PWRITE64
#endif

#if SANITIZER_INTERCEPT_WRITEV
INTERCEPTOR_WITH_SUFFIX(SSIZE_T, writev, int fd, __sanitizer_iovec *iov,
                        int iovcnt) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, writev, fd, iov, iovcnt);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(writev)(fd, iov, iovcnt);
  if (res > 0) read_iovec(ctx, iov, iovcnt, res);
  return res;
}
#define INIT_WRITEV INTERCEPT_FUNCTION(writev)
#else
#define INIT_WRITEV
#endif

#if SANITIZER_INTERCEPT_PWRITEV
INTERCEPTOR(SSIZE_T, pwritev, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwritev, fd, iov, iovcnt, offset);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwritev)(fd, iov, iovcnt, offset);
  if (res > 0) read_iovec(ctx, iov, iovcnt, res);
  return res;
}
#define INIT_PWRITEV INTERCEPT_FUNCTION(pwritev)
#else
#define INIT_PWRITEV
#endif

#if SANITIZER_INTERCEPT_PWRITEV64
INTERCEPTOR(SSIZE_T, pwritev64, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwritev64, fd, iov, iovcnt, offset);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwritev64)(fd, iov, iovcnt, offset);
  if (res > 0) read_iovec(ctx, iov, iovcnt, res);
  return res;
}
#define INIT_PWRITEV64 INTERCEPT_FUNCTION(pwritev64)
#else
#define INIT_PWRITEV64
#endif

#if SANITIZER_INTERCEPT_PRCTL
INTERCEPTOR(int, prctl, int option,
            unsigned long arg2, unsigned long arg3,   // NOLINT
            unsigned long arg4, unsigned long arg5) { // NOLINT
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, prctl, option, arg2, arg3, arg4, arg5);
  static const int PR_SET_NAME = 15;
  int res = REAL(prctl(option, arg2, arg3, arg4, arg5));
  if (option == PR_SET_NAME) {
    char buff[16];
    internal_strncpy(buff, (char *)arg2, 15);
    buff[15] = 0;
    COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, buff);
  }
  return res;
}
#define INIT_PRCTL INTERCEPT_FUNCTION(prctl)
#else
#define INIT_PRCTL
#endif // SANITIZER_INTERCEPT_PRCTL


#if SANITIZER_INTERCEPT_TIME
INTERCEPTOR(unsigned long, time, unsigned long *t) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, time, t);
  unsigned long res = REAL(time)(t);
  if (t && res != (unsigned long)-1) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, t, sizeof(*t));
  }
  return res;
}
#define INIT_TIME                                \
  INTERCEPT_FUNCTION(time);
#else
#define INIT_TIME
#endif // SANITIZER_INTERCEPT_TIME


#if SANITIZER_INTERCEPT_LOCALTIME_AND_FRIENDS
INTERCEPTOR(void *, localtime, unsigned long *timep) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, localtime, timep);
  void *res = REAL(localtime)(timep);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_tm_sz);
  }
  return res;
}
INTERCEPTOR(void *, localtime_r, unsigned long *timep, void *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, localtime_r, timep, result);
  void *res = REAL(localtime_r)(timep, result);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_tm_sz);
  }
  return res;
}
INTERCEPTOR(void *, gmtime, unsigned long *timep) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gmtime, timep);
  void *res = REAL(gmtime)(timep);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_tm_sz);
  }
  return res;
}
INTERCEPTOR(void *, gmtime_r, unsigned long *timep, void *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gmtime_r, timep, result);
  void *res = REAL(gmtime_r)(timep, result);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_tm_sz);
  }
  return res;
}
INTERCEPTOR(char *, ctime, unsigned long *timep) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ctime, timep);
  char *res = REAL(ctime)(timep);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
INTERCEPTOR(char *, ctime_r, unsigned long *timep, char *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ctime_r, timep, result);
  char *res = REAL(ctime_r)(timep, result);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
INTERCEPTOR(char *, asctime, void *tm) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, asctime, tm);
  char *res = REAL(asctime)(tm);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, tm, struct_tm_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
INTERCEPTOR(char *, asctime_r, void *tm, char *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, asctime_r, tm, result);
  char *res = REAL(asctime_r)(tm, result);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, tm, struct_tm_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
#define INIT_LOCALTIME_AND_FRIENDS               \
  INTERCEPT_FUNCTION(localtime);                 \
  INTERCEPT_FUNCTION(localtime_r);               \
  INTERCEPT_FUNCTION(gmtime);                    \
  INTERCEPT_FUNCTION(gmtime_r);                  \
  INTERCEPT_FUNCTION(ctime);                     \
  INTERCEPT_FUNCTION(ctime_r);                   \
  INTERCEPT_FUNCTION(asctime);                   \
  INTERCEPT_FUNCTION(asctime_r);
#else
#define INIT_LOCALTIME_AND_FRIENDS
#endif // SANITIZER_INTERCEPT_LOCALTIME_AND_FRIENDS

#if SANITIZER_INTERCEPT_SCANF

#include "sanitizer_common_interceptors_scanf.inc"

#define VSCANF_INTERCEPTOR_IMPL(vname, allowGnuMalloc, ...)                    \
  {                                                                            \
    void *ctx;                                                                 \
    COMMON_INTERCEPTOR_ENTER(ctx, vname, __VA_ARGS__);                         \
    va_list aq;                                                                \
    va_copy(aq, ap);                                                           \
    int res = REAL(vname)(__VA_ARGS__);                                        \
    if (res > 0)                                                               \
      scanf_common(ctx, res, allowGnuMalloc, format, aq);                      \
    va_end(aq);                                                                \
    return res;                                                                \
  }

INTERCEPTOR(int, vscanf, const char *format, va_list ap)
VSCANF_INTERCEPTOR_IMPL(vscanf, true, format, ap)

INTERCEPTOR(int, vsscanf, const char *str, const char *format, va_list ap)
VSCANF_INTERCEPTOR_IMPL(vsscanf, true, str, format, ap)

INTERCEPTOR(int, vfscanf, void *stream, const char *format, va_list ap)
VSCANF_INTERCEPTOR_IMPL(vfscanf, true, stream, format, ap)

#if SANITIZER_INTERCEPT_ISOC99_SCANF
INTERCEPTOR(int, __isoc99_vscanf, const char *format, va_list ap)
VSCANF_INTERCEPTOR_IMPL(__isoc99_vscanf, false, format, ap)

INTERCEPTOR(int, __isoc99_vsscanf, const char *str, const char *format,
            va_list ap)
VSCANF_INTERCEPTOR_IMPL(__isoc99_vsscanf, false, str, format, ap)

INTERCEPTOR(int, __isoc99_vfscanf, void *stream, const char *format, va_list ap)
VSCANF_INTERCEPTOR_IMPL(__isoc99_vfscanf, false, stream, format, ap)
#endif  // SANITIZER_INTERCEPT_ISOC99_SCANF

#define SCANF_INTERCEPTOR_IMPL(name, vname, ...)                               \
  {                                                                            \
    void *ctx;                                                                 \
    COMMON_INTERCEPTOR_ENTER(ctx, name, __VA_ARGS__);                          \
    va_list ap;                                                                \
    va_start(ap, format);                                                      \
    int res = vname(__VA_ARGS__, ap);                                          \
    va_end(ap);                                                                \
    return res;                                                                \
  }

INTERCEPTOR(int, scanf, const char *format, ...)
SCANF_INTERCEPTOR_IMPL(scanf, vscanf, format)

INTERCEPTOR(int, fscanf, void *stream, const char *format, ...)
SCANF_INTERCEPTOR_IMPL(fscanf, vfscanf, stream, format)

INTERCEPTOR(int, sscanf, const char *str, const char *format, ...)
SCANF_INTERCEPTOR_IMPL(sscanf, vsscanf, str, format)

#if SANITIZER_INTERCEPT_ISOC99_SCANF
INTERCEPTOR(int, __isoc99_scanf, const char *format, ...)
SCANF_INTERCEPTOR_IMPL(__isoc99_scanf, __isoc99_vscanf, format)

INTERCEPTOR(int, __isoc99_fscanf, void *stream, const char *format, ...)
SCANF_INTERCEPTOR_IMPL(__isoc99_fscanf, __isoc99_vfscanf, stream, format)

INTERCEPTOR(int, __isoc99_sscanf, const char *str, const char *format, ...)
SCANF_INTERCEPTOR_IMPL(__isoc99_sscanf, __isoc99_vsscanf, str, format)
#endif

#define INIT_SCANF                                                             \
  INTERCEPT_FUNCTION(scanf);                                                   \
  INTERCEPT_FUNCTION(sscanf);                                                  \
  INTERCEPT_FUNCTION(fscanf);                                                  \
  INTERCEPT_FUNCTION(vscanf);                                                  \
  INTERCEPT_FUNCTION(vsscanf);                                                 \
  INTERCEPT_FUNCTION(vfscanf);                                                 \
  INTERCEPT_FUNCTION(__isoc99_scanf);                                          \
  INTERCEPT_FUNCTION(__isoc99_sscanf);                                         \
  INTERCEPT_FUNCTION(__isoc99_fscanf);                                         \
  INTERCEPT_FUNCTION(__isoc99_vscanf);                                         \
  INTERCEPT_FUNCTION(__isoc99_vsscanf);                                        \
  INTERCEPT_FUNCTION(__isoc99_vfscanf);

#else
#define INIT_SCANF
#endif


#if SANITIZER_INTERCEPT_IOCTL
#include "sanitizer_common_interceptors_ioctl.inc"
INTERCEPTOR(int, ioctl, int d, unsigned request, void *arg) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ioctl, d, request, arg);

  CHECK(ioctl_initialized);

  // Note: TSan does not use common flags, and they are zero-initialized.
  // This effectively disables ioctl handling in TSan.
  if (!common_flags()->handle_ioctl)
    return REAL(ioctl)(d, request, arg);

  const ioctl_desc *desc = ioctl_lookup(request);
  if (!desc)
    Printf("WARNING: unknown ioctl %x\n", request);

  if (desc)
    ioctl_common_pre(ctx, desc, d, request, arg);
  int res = REAL(ioctl)(d, request, arg);
  // FIXME: some ioctls have different return values for success and failure.
  if (desc && res != -1)
    ioctl_common_post(ctx, desc, res, d, request, arg);
  return res;
}
#define INIT_IOCTL \
  ioctl_init();    \
  INTERCEPT_FUNCTION(ioctl);
#else
#define INIT_IOCTL
#endif


#if SANITIZER_INTERCEPT_GETPWNAM_AND_FRIENDS
INTERCEPTOR(void *, getpwnam, const char *name) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpwnam, name);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  void *res = REAL(getpwnam)(name);
  if (res != 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_passwd_sz);
  return res;
}
INTERCEPTOR(void *, getpwuid, u32 uid) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpwuid, uid);
  void *res = REAL(getpwuid)(uid);
  if (res != 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_passwd_sz);
  return res;
}
INTERCEPTOR(void *, getgrnam, const char *name) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgrnam, name);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  void *res = REAL(getgrnam)(name);
  if (res != 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_group_sz);
  return res;
}
INTERCEPTOR(void *, getgrgid, u32 gid) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgrgid, gid);
  void *res = REAL(getgrgid)(gid);
  if (res != 0)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_group_sz);
  return res;
}
#define INIT_GETPWNAM_AND_FRIENDS                  \
  INTERCEPT_FUNCTION(getpwnam);                    \
  INTERCEPT_FUNCTION(getpwuid);                    \
  INTERCEPT_FUNCTION(getgrnam);                    \
  INTERCEPT_FUNCTION(getgrgid);
#else
#define INIT_GETPWNAM_AND_FRIENDS
#endif


#if SANITIZER_INTERCEPT_GETPWNAM_R_AND_FRIENDS
INTERCEPTOR(int, getpwnam_r, const char *name, void *pwd,
    char *buf, SIZE_T buflen, void **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpwnam_r, name, pwd, buf, buflen, result);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  int res = REAL(getpwnam_r)(name, pwd, buf, buflen, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, pwd, struct_passwd_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, buflen);
  }
  return res;
}
INTERCEPTOR(int, getpwuid_r, u32 uid, void *pwd,
    char *buf, SIZE_T buflen, void **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpwuid_r, uid, pwd, buf, buflen, result);
  int res = REAL(getpwuid_r)(uid, pwd, buf, buflen, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, pwd, struct_passwd_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, buflen);
  }
  return res;
}
INTERCEPTOR(int, getgrnam_r, const char *name, void *grp,
    char *buf, SIZE_T buflen, void **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgrnam_r, name, grp, buf, buflen, result);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  int res = REAL(getgrnam_r)(name, grp, buf, buflen, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, grp, struct_group_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, buflen);
  }
  return res;
}
INTERCEPTOR(int, getgrgid_r, u32 gid, void *grp,
    char *buf, SIZE_T buflen, void **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgrgid_r, gid, grp, buf, buflen, result);
  int res = REAL(getgrgid_r)(gid, grp, buf, buflen, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, grp, struct_group_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, buflen);
  }
  return res;
}
#define INIT_GETPWNAM_R_AND_FRIENDS                \
  INTERCEPT_FUNCTION(getpwnam_r);                  \
  INTERCEPT_FUNCTION(getpwuid_r);                  \
  INTERCEPT_FUNCTION(getgrnam_r);                  \
  INTERCEPT_FUNCTION(getgrgid_r);
#else
#define INIT_GETPWNAM_R_AND_FRIENDS
#endif


#if SANITIZER_INTERCEPT_CLOCK_GETTIME
INTERCEPTOR(int, clock_getres, u32 clk_id, void *tp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, clock_getres, clk_id, tp);
  int res = REAL(clock_getres)(clk_id, tp);
  if (!res && tp) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, tp, struct_timespec_sz);
  }
  return res;
}
INTERCEPTOR(int, clock_gettime, u32 clk_id, void *tp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, clock_gettime, clk_id, tp);
  int res = REAL(clock_gettime)(clk_id, tp);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, tp, struct_timespec_sz);
  }
  return res;
}
INTERCEPTOR(int, clock_settime, u32 clk_id, const void *tp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, clock_settime, clk_id, tp);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, tp, struct_timespec_sz);
  return REAL(clock_settime)(clk_id, tp);
}
#define INIT_CLOCK_GETTIME                         \
  INTERCEPT_FUNCTION(clock_getres);                \
  INTERCEPT_FUNCTION(clock_gettime);               \
  INTERCEPT_FUNCTION(clock_settime);
#else
#define INIT_CLOCK_GETTIME
#endif


#if SANITIZER_INTERCEPT_GETITIMER
INTERCEPTOR(int, getitimer, int which, void *curr_value) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getitimer, which, curr_value);
  int res = REAL(getitimer)(which, curr_value);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, curr_value, struct_itimerval_sz);
  }
  return res;
}
INTERCEPTOR(int, setitimer, int which, const void *new_value, void *old_value) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, setitimer, which, new_value, old_value);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, new_value, struct_itimerval_sz);
  int res = REAL(setitimer)(which, new_value, old_value);
  if (!res && old_value) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, old_value, struct_itimerval_sz);
  }
  return res;
}
#define INIT_GETITIMER                             \
  INTERCEPT_FUNCTION(getitimer);                   \
  INTERCEPT_FUNCTION(setitimer);
#else
#define INIT_GETITIMER
#endif


#if SANITIZER_INTERCEPT_GLOB
struct sanitizer_glob_t {
  SIZE_T gl_pathc;
  char **gl_pathv;
};

static void unpoison_glob_t(void *ctx, sanitizer_glob_t *pglob) {
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, pglob, sizeof(*pglob));
  // +1 for NULL pointer at the end.
  COMMON_INTERCEPTOR_WRITE_RANGE(
      ctx, pglob->gl_pathv, (pglob->gl_pathc + 1) * sizeof(*pglob->gl_pathv));
  for (SIZE_T i = 0; i < pglob->gl_pathc; ++i) {
    char *p = pglob->gl_pathv[i];
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p, REAL(strlen)(p) + 1);
  }
}

INTERCEPTOR(int, glob, const char *pattern, int flags,
            int (*errfunc)(const char *epath, int eerrno),
            sanitizer_glob_t *pglob) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, glob, pattern, flags, errfunc, pglob);
  int res = REAL(glob)(pattern, flags, errfunc, pglob);
  if (res == 0)
    unpoison_glob_t(ctx, pglob);
  return res;
}

INTERCEPTOR(int, glob64, const char *pattern, int flags,
            int (*errfunc)(const char *epath, int eerrno),
            sanitizer_glob_t *pglob) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, glob64, pattern, flags, errfunc, pglob);
  int res = REAL(glob64)(pattern, flags, errfunc, pglob);
  if (res == 0)
    unpoison_glob_t(ctx, pglob);
  return res;
}
#define INIT_GLOB                               \
  INTERCEPT_FUNCTION(glob);                     \
  INTERCEPT_FUNCTION(glob64);
#else // SANITIZER_INTERCEPT_GLOB
#define INIT_GLOB
#endif // SANITIZER_INTERCEPT_GLOB


#if SANITIZER_INTERCEPT_WAIT
// According to sys/wait.h, wait(), waitid(), waitpid() may have symbol version
// suffixes on Darwin. See the declaration of INTERCEPTOR_WITH_SUFFIX for
// details.
INTERCEPTOR_WITH_SUFFIX(int, wait, int *status) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wait, status);
  int res = REAL(wait)(status);
  if (res != -1 && status)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, status, sizeof(*status));
  return res;
}
INTERCEPTOR_WITH_SUFFIX(int, waitid, int idtype, int id, void *infop,
  int options) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, waitid, idtype, id, infop, options);
  int res = REAL(waitid)(idtype, id, infop, options);
  if (res != -1 && infop)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, infop, siginfo_t_sz);
  return res;
}
INTERCEPTOR_WITH_SUFFIX(int, waitpid, int pid, int *status, int options) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, waitpid, pid, status, options);
  int res = REAL(waitpid)(pid, status, options);
  if (res != -1 && status)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, status, sizeof(*status));
  return res;
}
INTERCEPTOR(int, wait3, int *status, int options, void *rusage) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wait3, status, options, rusage);
  int res = REAL(wait3)(status, options, rusage);
  if (res != -1) {
    if (status)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, status, sizeof(*status));
    if (rusage)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, rusage, struct_rusage_sz);
  }
  return res;
}
INTERCEPTOR(int, wait4, int pid, int *status, int options, void *rusage) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wait4, pid, status, options, rusage);
  int res = REAL(wait4)(pid, status, options, rusage);
  if (res != -1) {
    if (status)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, status, sizeof(*status));
    if (rusage)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, rusage, struct_rusage_sz);
  }
  return res;
}
#define INIT_WAIT                                \
  INTERCEPT_FUNCTION(wait);                      \
  INTERCEPT_FUNCTION(waitid);                    \
  INTERCEPT_FUNCTION(waitpid);                   \
  INTERCEPT_FUNCTION(wait3);                     \
  INTERCEPT_FUNCTION(wait4);
#else
#define INIT_WAIT
#endif

#if SANITIZER_INTERCEPT_INET
INTERCEPTOR(char *, inet_ntop, int af, const void *src, char *dst, u32 size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, inet_ntop, af, src, dst, size);
  uptr sz = __sanitizer_in_addr_sz(af);
  if (sz) COMMON_INTERCEPTOR_READ_RANGE(ctx, src, sz);
  // FIXME: figure out read size based on the address family.
  char *res = REAL(inet_ntop)(af, src, dst, size);
  if (res)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
INTERCEPTOR(int, inet_pton, int af, const char *src, void *dst) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, inet_pton, af, src, dst);
  // FIXME: figure out read size based on the address family.
  int res = REAL(inet_pton)(af, src, dst);
  if (res == 1) {
    uptr sz = __sanitizer_in_addr_sz(af);
    if (sz) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dst, sz);
  }
  return res;
}
#define INIT_INET                                \
  INTERCEPT_FUNCTION(inet_ntop);                 \
  INTERCEPT_FUNCTION(inet_pton);
#else
#define INIT_INET
#endif

#if SANITIZER_INTERCEPT_INET
INTERCEPTOR(int, inet_aton, const char *cp, void *dst) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, inet_aton, cp, dst);
  if (cp) COMMON_INTERCEPTOR_READ_RANGE(ctx, cp, REAL(strlen)(cp) + 1);
  int res = REAL(inet_aton)(cp, dst);
  if (res != 0) {
    uptr sz = __sanitizer_in_addr_sz(af_inet);
    if (sz) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dst, sz);
  }
  return res;
}
#define INIT_INET_ATON INTERCEPT_FUNCTION(inet_aton);
#else
#define INIT_INET_ATON
#endif

#if SANITIZER_INTERCEPT_PTHREAD_GETSCHEDPARAM
INTERCEPTOR(int, pthread_getschedparam, uptr thread, int *policy, int *param) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_getschedparam, thread, policy, param);
  int res = REAL(pthread_getschedparam)(thread, policy, param);
  if (res == 0) {
    if (policy) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, policy, sizeof(*policy));
    if (param) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, param, sizeof(*param));
  }
  return res;
}
#define INIT_PTHREAD_GETSCHEDPARAM INTERCEPT_FUNCTION(pthread_getschedparam);
#else
#define INIT_PTHREAD_GETSCHEDPARAM
#endif

#if SANITIZER_INTERCEPT_GETADDRINFO
INTERCEPTOR(int, getaddrinfo, char *node, char *service,
            struct __sanitizer_addrinfo *hints,
            struct __sanitizer_addrinfo **out) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getaddrinfo, node, service, hints, out);
  if (node) COMMON_INTERCEPTOR_READ_RANGE(ctx, node, REAL(strlen)(node) + 1);
  if (service)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, service, REAL(strlen)(service) + 1);
  if (hints)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, hints, sizeof(__sanitizer_addrinfo));
  int res = REAL(getaddrinfo)(node, service, hints, out);
  if (res == 0 && out) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, out, sizeof(*out));
    struct __sanitizer_addrinfo *p = *out;
    while (p) {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p, sizeof(*p));
      if (p->ai_addr)
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p->ai_addr, p->ai_addrlen);
      if (p->ai_canonname)
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p->ai_canonname,
                                       REAL(strlen)(p->ai_canonname) + 1);
      p = p->ai_next;
    }
  }
  return res;
}
#define INIT_GETADDRINFO INTERCEPT_FUNCTION(getaddrinfo);
#else
#define INIT_GETADDRINFO
#endif

#if SANITIZER_INTERCEPT_GETSOCKNAME
INTERCEPTOR(int, getsockname, int sock_fd, void *addr, int *addrlen) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getsockname, sock_fd, addr, addrlen);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, addrlen, sizeof(*addrlen));
  int addrlen_in = *addrlen;
  int res = REAL(getsockname)(sock_fd, addr, addrlen);
  if (res == 0) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, Min(addrlen_in, *addrlen));
  }
  return res;
}
#define INIT_GETSOCKNAME INTERCEPT_FUNCTION(getsockname);
#else
#define INIT_GETSOCKNAME
#endif

#if SANITIZER_INTERCEPT_GETHOSTBYNAME || SANITIZER_INTERCEPT_GETHOSTBYNAME_R
static void write_hostent(void *ctx, struct __sanitizer_hostent *h) {
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h, sizeof(__sanitizer_hostent));
  if (h->h_name)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h->h_name, REAL(strlen)(h->h_name) + 1);
  char **p = h->h_aliases;
  while (*p) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *p, REAL(strlen)(*p) + 1);
    ++p;
  }
  COMMON_INTERCEPTOR_WRITE_RANGE(
      ctx, h->h_aliases, (p - h->h_aliases + 1) * sizeof(*h->h_aliases));
  p = h->h_addr_list;
  while (*p) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *p, h->h_length);
    ++p;
  }
  COMMON_INTERCEPTOR_WRITE_RANGE(
      ctx, h->h_addr_list, (p - h->h_addr_list + 1) * sizeof(*h->h_addr_list));
}
#endif

#if SANITIZER_INTERCEPT_GETHOSTBYNAME
INTERCEPTOR(struct __sanitizer_hostent *, gethostbyname, char *name) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostbyname, name);
  struct __sanitizer_hostent *res = REAL(gethostbyname)(name);
  if (res) write_hostent(ctx, res);
  return res;
}

INTERCEPTOR(struct __sanitizer_hostent *, gethostbyaddr, void *addr, int len,
            int type) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostbyaddr, addr, len, type);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, addr, len);
  struct __sanitizer_hostent *res = REAL(gethostbyaddr)(addr, len, type);
  if (res) write_hostent(ctx, res);
  return res;
}

INTERCEPTOR(struct __sanitizer_hostent *, gethostent) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostent);
  struct __sanitizer_hostent *res = REAL(gethostent)();
  if (res) write_hostent(ctx, res);
  return res;
}

INTERCEPTOR(struct __sanitizer_hostent *, gethostbyname2, char *name, int af) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostbyname2, name, af);
  struct __sanitizer_hostent *res = REAL(gethostbyname2)(name, af);
  if (res) write_hostent(ctx, res);
  return res;
}
#define INIT_GETHOSTBYNAME           \
  INTERCEPT_FUNCTION(gethostent);    \
  INTERCEPT_FUNCTION(gethostbyaddr); \
  INTERCEPT_FUNCTION(gethostbyname); \
  INTERCEPT_FUNCTION(gethostbyname2);
#else
#define INIT_GETHOSTBYNAME
#endif

#if SANITIZER_INTERCEPT_GETHOSTBYNAME_R
INTERCEPTOR(int, gethostent_r, struct __sanitizer_hostent *ret, char *buf,
            SIZE_T buflen, __sanitizer_hostent **result, int *h_errnop) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostent_r, ret, buf, buflen, result,
                           h_errnop);
  int res = REAL(gethostent_r)(ret, buf, buflen, result, h_errnop);
  if (res == 0) {
    if (result) {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
      if (*result) write_hostent(ctx, *result);
    }
    if (h_errnop)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
  }
  return res;
}

INTERCEPTOR(int, gethostbyaddr_r, void *addr, int len, int type,
            struct __sanitizer_hostent *ret, char *buf, SIZE_T buflen,
            __sanitizer_hostent **result, int *h_errnop) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostbyaddr_r, addr, len, type, ret, buf,
                           buflen, result, h_errnop);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, addr, len);
  int res = REAL(gethostbyaddr_r)(addr, len, type, ret, buf, buflen, result,
                                  h_errnop);
  if (res == 0) {
    if (result) {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
      if (*result) write_hostent(ctx, *result);
    }
    if (h_errnop)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
  }
  return res;
}

INTERCEPTOR(int, gethostbyname_r, char *name, struct __sanitizer_hostent *ret,
            char *buf, SIZE_T buflen, __sanitizer_hostent **result,
            int *h_errnop) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostbyname_r, name, ret, buf, buflen, result,
                           h_errnop);
  int res = REAL(gethostbyname_r)(name, ret, buf, buflen, result, h_errnop);
  if (res == 0) {
    if (result) {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
      if (*result) write_hostent(ctx, *result);
    }
    if (h_errnop)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
  }
  return res;
}

INTERCEPTOR(int, gethostbyname2_r, char *name, int af,
            struct __sanitizer_hostent *ret, char *buf, SIZE_T buflen,
            __sanitizer_hostent **result, int *h_errnop) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostbyname2_r, name, af, ret, buf, buflen,
                           result, h_errnop);
  int res =
      REAL(gethostbyname2_r)(name, af, ret, buf, buflen, result, h_errnop);
  if (res == 0) {
    if (result) {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
      if (*result) write_hostent(ctx, *result);
    }
    if (h_errnop)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
  }
  return res;
}
#define INIT_GETHOSTBYNAME_R           \
  INTERCEPT_FUNCTION(gethostent_r);    \
  INTERCEPT_FUNCTION(gethostbyaddr_r); \
  INTERCEPT_FUNCTION(gethostbyname_r); \
  INTERCEPT_FUNCTION(gethostbyname2_r);
#else
#define INIT_GETHOSTBYNAME_R
#endif

#if SANITIZER_INTERCEPT_GETSOCKOPT
INTERCEPTOR(int, getsockopt, int sockfd, int level, int optname, void *optval,
            int *optlen) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getsockopt, sockfd, level, optname, optval,
                           optlen);
  if (optlen) COMMON_INTERCEPTOR_READ_RANGE(ctx, optlen, sizeof(*optlen));
  int res = REAL(getsockopt)(sockfd, level, optname, optval, optlen);
  if (res == 0)
    if (optval && optlen) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, optval, *optlen);
  return res;
}
#define INIT_GETSOCKOPT INTERCEPT_FUNCTION(getsockopt);
#else
#define INIT_GETSOCKOPT
#endif

#if SANITIZER_INTERCEPT_ACCEPT
INTERCEPTOR(int, accept, int fd, void *addr, unsigned *addrlen) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, accept, fd, addr, addrlen);
  unsigned addrlen0;
  if (addrlen) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, addrlen, sizeof(*addrlen));
    addrlen0 = *addrlen;
  }
  int fd2 = REAL(accept)(fd, addr, addrlen);
  if (fd2 >= 0) {
    if (fd >= 0)
      COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, fd2);
    if (addr && addrlen)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, Min(*addrlen, addrlen0));
  }
  return fd2;
}
#define INIT_ACCEPT INTERCEPT_FUNCTION(accept);
#else
#define INIT_ACCEPT
#endif

#if SANITIZER_INTERCEPT_ACCEPT4
INTERCEPTOR(int, accept4, int fd, void *addr, unsigned *addrlen, int f) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, accept4, fd, addr, addrlen, f);
  unsigned addrlen0;
  if (addrlen) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, addrlen, sizeof(*addrlen));
    addrlen0 = *addrlen;
  }
  int fd2 = REAL(accept4)(fd, addr, addrlen, f);
  if (fd2 >= 0) {
    if (fd >= 0)
      COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, fd2);
    if (addr && addrlen)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, Min(*addrlen, addrlen0));
  }
  return fd2;
}
#define INIT_ACCEPT4 INTERCEPT_FUNCTION(accept4);
#else
#define INIT_ACCEPT4
#endif

#if SANITIZER_INTERCEPT_MODF
INTERCEPTOR(double, modf, double x, double *iptr) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, modf, x, iptr);
  double res = REAL(modf)(x, iptr);
  if (iptr) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, iptr, sizeof(*iptr));
  }
  return res;
}
INTERCEPTOR(float, modff, float x, float *iptr) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, modff, x, iptr);
  float res = REAL(modff)(x, iptr);
  if (iptr) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, iptr, sizeof(*iptr));
  }
  return res;
}
INTERCEPTOR(long double, modfl, long double x, long double *iptr) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, modfl, x, iptr);
  long double res = REAL(modfl)(x, iptr);
  if (iptr) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, iptr, sizeof(*iptr));
  }
  return res;
}
#define INIT_MODF            \
  INTERCEPT_FUNCTION(modf);  \
  INTERCEPT_FUNCTION(modff); \
  INTERCEPT_FUNCTION(modfl);
#else
#define INIT_MODF
#endif

#if SANITIZER_INTERCEPT_RECVMSG
static void write_msghdr(void *ctx, struct __sanitizer_msghdr *msg,
                         SSIZE_T maxlen) {
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg, sizeof(*msg));
  if (msg->msg_name)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg->msg_name,
                                   REAL(strlen)((char *)msg->msg_name) + 1);
  if (msg->msg_iov)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg->msg_iov,
                                   sizeof(*msg->msg_iov) * msg->msg_iovlen);
  write_iovec(ctx, msg->msg_iov, msg->msg_iovlen, maxlen);
  if (msg->msg_control)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg->msg_control, msg->msg_controllen);
}

INTERCEPTOR(SSIZE_T, recvmsg, int fd, struct __sanitizer_msghdr *msg,
            int flags) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, recvmsg, fd, msg, flags);
  SSIZE_T res = REAL(recvmsg)(fd, msg, flags);
  if (res >= 0) {
    if (fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
    if (msg) write_msghdr(ctx, msg, res);
  }
  return res;
}
#define INIT_RECVMSG INTERCEPT_FUNCTION(recvmsg);
#else
#define INIT_RECVMSG
#endif

#if SANITIZER_INTERCEPT_GETPEERNAME
INTERCEPTOR(int, getpeername, int sockfd, void *addr, unsigned *addrlen) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpeername, sockfd, addr, addrlen);
  unsigned addr_sz;
  if (addrlen) addr_sz = *addrlen;
  int res = REAL(getpeername)(sockfd, addr, addrlen);
  if (!res && addr && addrlen)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, Min(addr_sz, *addrlen));
  return res;
}
#define INIT_GETPEERNAME INTERCEPT_FUNCTION(getpeername);
#else
#define INIT_GETPEERNAME
#endif

#if SANITIZER_INTERCEPT_SYSINFO
INTERCEPTOR(int, sysinfo, void *info) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sysinfo, info);
  int res = REAL(sysinfo)(info);
  if (!res && info)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, info, struct_sysinfo_sz);
  return res;
}
#define INIT_SYSINFO INTERCEPT_FUNCTION(sysinfo);
#else
#define INIT_SYSINFO
#endif

#if SANITIZER_INTERCEPT_READDIR
INTERCEPTOR(__sanitizer_dirent *, readdir, void *dirp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir, dirp);
  __sanitizer_dirent *res = REAL(readdir)(dirp);
  if (res)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, res->d_reclen);
  return res;
}

INTERCEPTOR(int, readdir_r, void *dirp, __sanitizer_dirent *entry, __sanitizer_dirent **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir_r, dirp, entry, result);
  int res = REAL(readdir_r)(dirp, entry, result);
  if (!res) {
    if (result)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (entry)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, entry, entry->d_reclen);
  }
  return res;
}

#define INIT_READDIR           \
  INTERCEPT_FUNCTION(readdir); \
  INTERCEPT_FUNCTION(readdir_r);
#else
#define INIT_READDIR
#endif

#if SANITIZER_INTERCEPT_READDIR64
INTERCEPTOR(__sanitizer_dirent64 *, readdir64, void *dirp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir64, dirp);
  __sanitizer_dirent64 *res = REAL(readdir64)(dirp);
  if (res)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, res->d_reclen);
  return res;
}

INTERCEPTOR(int, readdir64_r, void *dirp, __sanitizer_dirent64 *entry, __sanitizer_dirent64 **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir64_r, dirp, entry, result);
  int res = REAL(readdir64_r)(dirp, entry, result);
  if (!res) {
    if (result)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (entry)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, entry, entry->d_reclen);
  }
  return res;
}
#define INIT_READDIR64           \
  INTERCEPT_FUNCTION(readdir64); \
  INTERCEPT_FUNCTION(readdir64_r);
#else
#define INIT_READDIR64
#endif


#define SANITIZER_COMMON_INTERCEPTORS_INIT \
  INIT_STRCASECMP;                         \
  INIT_STRNCASECMP;                        \
  INIT_READ;                               \
  INIT_PREAD;                              \
  INIT_PREAD64;                            \
  INIT_READV;                              \
  INIT_PREADV;                             \
  INIT_PREADV64;                           \
  INIT_WRITE;                              \
  INIT_PWRITE;                             \
  INIT_PWRITE64;                           \
  INIT_WRITEV;                             \
  INIT_PWRITEV;                            \
  INIT_PWRITEV64;                          \
  INIT_PRCTL;                              \
  INIT_LOCALTIME_AND_FRIENDS;              \
  INIT_SCANF;                              \
  INIT_FREXP;                              \
  INIT_FREXPF_FREXPL;                      \
  INIT_GETPWNAM_AND_FRIENDS;               \
  INIT_GETPWNAM_R_AND_FRIENDS;             \
  INIT_CLOCK_GETTIME;                      \
  INIT_GETITIMER;                          \
  INIT_TIME;                               \
  INIT_GLOB;                               \
  INIT_WAIT;                               \
  INIT_INET;                               \
  INIT_PTHREAD_GETSCHEDPARAM;              \
  INIT_GETADDRINFO;                        \
  INIT_GETSOCKNAME;                        \
  INIT_GETHOSTBYNAME;                      \
  INIT_GETHOSTBYNAME_R;                    \
  INIT_GETSOCKOPT;                         \
  INIT_ACCEPT;                             \
  INIT_ACCEPT4;                            \
  INIT_MODF;                               \
  INIT_RECVMSG;                            \
  INIT_GETPEERNAME;                        \
  INIT_IOCTL;                              \
  INIT_INET_ATON;                          \
  INIT_SYSINFO;                            \
  INIT_READDIR;                            \
  INIT_READDIR64;
