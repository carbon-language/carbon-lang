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
//   COMMON_INTERCEPTOR_INITIALIZE_RANGE
//   COMMON_INTERCEPTOR_FD_ACQUIRE
//   COMMON_INTERCEPTOR_FD_RELEASE
//   COMMON_INTERCEPTOR_FD_ACCESS
//   COMMON_INTERCEPTOR_SET_THREAD_NAME
//   COMMON_INTERCEPTOR_ON_EXIT
//   COMMON_INTERCEPTOR_MUTEX_LOCK
//   COMMON_INTERCEPTOR_MUTEX_UNLOCK
//   COMMON_INTERCEPTOR_MUTEX_REPAIR
//   COMMON_INTERCEPTOR_SET_PTHREAD_NAME
//   COMMON_INTERCEPTOR_HANDLE_RECVMSG
//===----------------------------------------------------------------------===//
#include "interception/interception.h"
#include "sanitizer_platform_interceptors.h"
#include "sanitizer_tls_get_addr.h"

#include <stdarg.h>

#if SANITIZER_WINDOWS && !defined(va_copy)
#define va_copy(dst, src) ((dst) = (src))
#endif // _WIN32

#ifndef COMMON_INTERCEPTOR_INITIALIZE_RANGE
#define COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, p, size) {}
#endif

#ifndef COMMON_INTERCEPTOR_FD_ACCESS
#define COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd) {}
#endif

#ifndef COMMON_INTERCEPTOR_MUTEX_LOCK
#define COMMON_INTERCEPTOR_MUTEX_LOCK(ctx, m) {}
#endif

#ifndef COMMON_INTERCEPTOR_MUTEX_UNLOCK
#define COMMON_INTERCEPTOR_MUTEX_UNLOCK(ctx, m) {}
#endif

#ifndef COMMON_INTERCEPTOR_MUTEX_REPAIR
#define COMMON_INTERCEPTOR_MUTEX_REPAIR(ctx, m) {}
#endif

#ifndef COMMON_INTERCEPTOR_HANDLE_RECVMSG
#define COMMON_INTERCEPTOR_HANDLE_RECVMSG(ctx, msg) ((void)(msg))
#endif

#if SANITIZER_INTERCEPT_TEXTDOMAIN
INTERCEPTOR(char*, textdomain, const char *domainname) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, textdomain, domainname);
  char* domain = REAL(textdomain)(domainname);
  if (domain) {
    COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, domain,
        REAL(strlen)(domain) + 1);
  }
  return domain;
}
#define INIT_TEXTDOMAIN COMMON_INTERCEPT_FUNCTION(textdomain)
#else
#define INIT_TEXTDOMAIN
#endif

#if SANITIZER_INTERCEPT_STRCMP
static inline int CharCmpX(unsigned char c1, unsigned char c2) {
  return (c1 == c2) ? 0 : (c1 < c2) ? -1 : 1;
}

INTERCEPTOR(int, strcmp, const char *s1, const char *s2) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strcmp, s1, s2);
  unsigned char c1, c2;
  uptr i;
  for (i = 0;; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s1, i + 1);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s2, i + 1);
  return CharCmpX(c1, c2);
}

INTERCEPTOR(int, strncmp, const char *s1, const char *s2, uptr size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strncmp, s1, s2, size);
  unsigned char c1 = 0, c2 = 0;
  uptr i;
  for (i = 0; i < size; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s1, Min(i + 1, size));
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s2, Min(i + 1, size));
  return CharCmpX(c1, c2);
}

#define INIT_STRCMP COMMON_INTERCEPT_FUNCTION(strcmp)
#define INIT_STRNCMP COMMON_INTERCEPT_FUNCTION(strncmp)
#else
#define INIT_STRCMP
#define INIT_STRNCMP
#endif

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
  for (i = 0;; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
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
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
  }
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s1, Min(i + 1, n));
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s2, Min(i + 1, n));
  return CharCaseCmp(c1, c2);
}

#define INIT_STRCASECMP COMMON_INTERCEPT_FUNCTION(strcasecmp)
#define INIT_STRNCASECMP COMMON_INTERCEPT_FUNCTION(strncasecmp)
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

#define INIT_FREXP COMMON_INTERCEPT_FUNCTION(frexp);
#else
#define INIT_FREXP
#endif  // SANITIZER_INTERCEPT_FREXP

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

#define INIT_FREXPF_FREXPL           \
  COMMON_INTERCEPT_FUNCTION(frexpf); \
  COMMON_INTERCEPT_FUNCTION(frexpl)
#else
#define INIT_FREXPF_FREXPL
#endif  // SANITIZER_INTERCEPT_FREXPF_FREXPL

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
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  SSIZE_T res = REAL(read)(fd, ptr, count);
  if (res > 0) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_READ COMMON_INTERCEPT_FUNCTION(read)
#else
#define INIT_READ
#endif

#if SANITIZER_INTERCEPT_PREAD
INTERCEPTOR(SSIZE_T, pread, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pread, fd, ptr, count, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  SSIZE_T res = REAL(pread)(fd, ptr, count, offset);
  if (res > 0) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREAD COMMON_INTERCEPT_FUNCTION(pread)
#else
#define INIT_PREAD
#endif

#if SANITIZER_INTERCEPT_PREAD64
INTERCEPTOR(SSIZE_T, pread64, int fd, void *ptr, SIZE_T count, OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pread64, fd, ptr, count, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  SSIZE_T res = REAL(pread64)(fd, ptr, count, offset);
  if (res > 0) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREAD64 COMMON_INTERCEPT_FUNCTION(pread64)
#else
#define INIT_PREAD64
#endif

#if SANITIZER_INTERCEPT_READV
INTERCEPTOR_WITH_SUFFIX(SSIZE_T, readv, int fd, __sanitizer_iovec *iov,
                        int iovcnt) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readv, fd, iov, iovcnt);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  SSIZE_T res = REAL(readv)(fd, iov, iovcnt);
  if (res > 0) write_iovec(ctx, iov, iovcnt, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_READV COMMON_INTERCEPT_FUNCTION(readv)
#else
#define INIT_READV
#endif

#if SANITIZER_INTERCEPT_PREADV
INTERCEPTOR(SSIZE_T, preadv, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, preadv, fd, iov, iovcnt, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  SSIZE_T res = REAL(preadv)(fd, iov, iovcnt, offset);
  if (res > 0) write_iovec(ctx, iov, iovcnt, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREADV COMMON_INTERCEPT_FUNCTION(preadv)
#else
#define INIT_PREADV
#endif

#if SANITIZER_INTERCEPT_PREADV64
INTERCEPTOR(SSIZE_T, preadv64, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, preadv64, fd, iov, iovcnt, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  SSIZE_T res = REAL(preadv64)(fd, iov, iovcnt, offset);
  if (res > 0) write_iovec(ctx, iov, iovcnt, res);
  if (res >= 0 && fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
  return res;
}
#define INIT_PREADV64 COMMON_INTERCEPT_FUNCTION(preadv64)
#else
#define INIT_PREADV64
#endif

#if SANITIZER_INTERCEPT_WRITE
INTERCEPTOR(SSIZE_T, write, int fd, void *ptr, SIZE_T count) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, write, fd, ptr, count);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(write)(fd, ptr, count);
  // FIXME: this check should be _before_ the call to REAL(write), not after
  if (res > 0) COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
#define INIT_WRITE COMMON_INTERCEPT_FUNCTION(write)
#else
#define INIT_WRITE
#endif

#if SANITIZER_INTERCEPT_PWRITE
INTERCEPTOR(SSIZE_T, pwrite, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwrite, fd, ptr, count, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwrite)(fd, ptr, count, offset);
  if (res > 0) COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
#define INIT_PWRITE COMMON_INTERCEPT_FUNCTION(pwrite)
#else
#define INIT_PWRITE
#endif

#if SANITIZER_INTERCEPT_PWRITE64
INTERCEPTOR(SSIZE_T, pwrite64, int fd, void *ptr, OFF64_T count,
            OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwrite64, fd, ptr, count, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwrite64)(fd, ptr, count, offset);
  if (res > 0) COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, res);
  return res;
}
#define INIT_PWRITE64 COMMON_INTERCEPT_FUNCTION(pwrite64)
#else
#define INIT_PWRITE64
#endif

#if SANITIZER_INTERCEPT_WRITEV
INTERCEPTOR_WITH_SUFFIX(SSIZE_T, writev, int fd, __sanitizer_iovec *iov,
                        int iovcnt) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, writev, fd, iov, iovcnt);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(writev)(fd, iov, iovcnt);
  if (res > 0) read_iovec(ctx, iov, iovcnt, res);
  return res;
}
#define INIT_WRITEV COMMON_INTERCEPT_FUNCTION(writev)
#else
#define INIT_WRITEV
#endif

#if SANITIZER_INTERCEPT_PWRITEV
INTERCEPTOR(SSIZE_T, pwritev, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwritev, fd, iov, iovcnt, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwritev)(fd, iov, iovcnt, offset);
  if (res > 0) read_iovec(ctx, iov, iovcnt, res);
  return res;
}
#define INIT_PWRITEV COMMON_INTERCEPT_FUNCTION(pwritev)
#else
#define INIT_PWRITEV
#endif

#if SANITIZER_INTERCEPT_PWRITEV64
INTERCEPTOR(SSIZE_T, pwritev64, int fd, __sanitizer_iovec *iov, int iovcnt,
            OFF64_T offset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pwritev64, fd, iov, iovcnt, offset);
  COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd);
  if (fd >= 0) COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd);
  SSIZE_T res = REAL(pwritev64)(fd, iov, iovcnt, offset);
  if (res > 0) read_iovec(ctx, iov, iovcnt, res);
  return res;
}
#define INIT_PWRITEV64 COMMON_INTERCEPT_FUNCTION(pwritev64)
#else
#define INIT_PWRITEV64
#endif

#if SANITIZER_INTERCEPT_PRCTL
INTERCEPTOR(int, prctl, int option, unsigned long arg2,
            unsigned long arg3,                        // NOLINT
            unsigned long arg4, unsigned long arg5) {  // NOLINT
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
#define INIT_PRCTL COMMON_INTERCEPT_FUNCTION(prctl)
#else
#define INIT_PRCTL
#endif  // SANITIZER_INTERCEPT_PRCTL

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
#define INIT_TIME COMMON_INTERCEPT_FUNCTION(time);
#else
#define INIT_TIME
#endif  // SANITIZER_INTERCEPT_TIME

#if SANITIZER_INTERCEPT_LOCALTIME_AND_FRIENDS
static void unpoison_tm(void *ctx, __sanitizer_tm *tm) {
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, tm, sizeof(*tm));
  if (tm->tm_zone) {
    // Can not use COMMON_INTERCEPTOR_WRITE_RANGE here, because tm->tm_zone
    // can point to shared memory and tsan would report a data race.
    COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, tm->tm_zone,
                                        REAL(strlen(tm->tm_zone)) + 1);
  }
}
INTERCEPTOR(__sanitizer_tm *, localtime, unsigned long *timep) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, localtime, timep);
  __sanitizer_tm *res = REAL(localtime)(timep);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    unpoison_tm(ctx, res);
  }
  return res;
}
INTERCEPTOR(__sanitizer_tm *, localtime_r, unsigned long *timep, void *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, localtime_r, timep, result);
  __sanitizer_tm *res = REAL(localtime_r)(timep, result);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    unpoison_tm(ctx, res);
  }
  return res;
}
INTERCEPTOR(__sanitizer_tm *, gmtime, unsigned long *timep) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gmtime, timep);
  __sanitizer_tm *res = REAL(gmtime)(timep);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    unpoison_tm(ctx, res);
  }
  return res;
}
INTERCEPTOR(__sanitizer_tm *, gmtime_r, unsigned long *timep, void *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gmtime_r, timep, result);
  __sanitizer_tm *res = REAL(gmtime_r)(timep, result);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timep, sizeof(*timep));
    unpoison_tm(ctx, res);
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
INTERCEPTOR(char *, asctime, __sanitizer_tm *tm) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, asctime, tm);
  char *res = REAL(asctime)(tm);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, tm, sizeof(*tm));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
INTERCEPTOR(char *, asctime_r, __sanitizer_tm *tm, char *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, asctime_r, tm, result);
  char *res = REAL(asctime_r)(tm, result);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, tm, sizeof(*tm));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
#define INIT_LOCALTIME_AND_FRIENDS        \
  COMMON_INTERCEPT_FUNCTION(localtime);   \
  COMMON_INTERCEPT_FUNCTION(localtime_r); \
  COMMON_INTERCEPT_FUNCTION(gmtime);      \
  COMMON_INTERCEPT_FUNCTION(gmtime_r);    \
  COMMON_INTERCEPT_FUNCTION(ctime);       \
  COMMON_INTERCEPT_FUNCTION(ctime_r);     \
  COMMON_INTERCEPT_FUNCTION(asctime);     \
  COMMON_INTERCEPT_FUNCTION(asctime_r);
#else
#define INIT_LOCALTIME_AND_FRIENDS
#endif  // SANITIZER_INTERCEPT_LOCALTIME_AND_FRIENDS

#if SANITIZER_INTERCEPT_STRPTIME
INTERCEPTOR(char *, strptime, char *s, char *format, __sanitizer_tm *tm) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strptime, s, format, tm);
  if (format)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, format, REAL(strlen)(format) + 1);
  char *res = REAL(strptime)(s, format, tm);
  if (res) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, s, res - s);
    // Do not call unpoison_tm here, because strptime does not, in fact,
    // initialize the entire struct tm. For example, tm_zone pointer is left
    // uninitialized.
    if (tm) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, tm, sizeof(*tm));
  }
  return res;
}
#define INIT_STRPTIME COMMON_INTERCEPT_FUNCTION(strptime);
#else
#define INIT_STRPTIME
#endif

#if SANITIZER_INTERCEPT_SCANF || SANITIZER_INTERCEPT_PRINTF
#include "sanitizer_common_interceptors_format.inc"

#define FORMAT_INTERCEPTOR_IMPL(name, vname, ...)                              \
  {                                                                            \
    void *ctx;                                                                 \
    va_list ap;                                                                \
    va_start(ap, format);                                                      \
    COMMON_INTERCEPTOR_ENTER(ctx, vname, __VA_ARGS__, ap);                     \
    int res = WRAP(vname)(__VA_ARGS__, ap);                                    \
    va_end(ap);                                                                \
    return res;                                                                \
  }

#endif

#if SANITIZER_INTERCEPT_SCANF

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

INTERCEPTOR(int, scanf, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(scanf, vscanf, format)

INTERCEPTOR(int, fscanf, void *stream, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(fscanf, vfscanf, stream, format)

INTERCEPTOR(int, sscanf, const char *str, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(sscanf, vsscanf, str, format)

#if SANITIZER_INTERCEPT_ISOC99_SCANF
INTERCEPTOR(int, __isoc99_scanf, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(__isoc99_scanf, __isoc99_vscanf, format)

INTERCEPTOR(int, __isoc99_fscanf, void *stream, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(__isoc99_fscanf, __isoc99_vfscanf, stream, format)

INTERCEPTOR(int, __isoc99_sscanf, const char *str, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(__isoc99_sscanf, __isoc99_vsscanf, str, format)
#endif

#endif

#if SANITIZER_INTERCEPT_SCANF
#define INIT_SCANF                    \
  COMMON_INTERCEPT_FUNCTION(scanf);   \
  COMMON_INTERCEPT_FUNCTION(sscanf);  \
  COMMON_INTERCEPT_FUNCTION(fscanf);  \
  COMMON_INTERCEPT_FUNCTION(vscanf);  \
  COMMON_INTERCEPT_FUNCTION(vsscanf); \
  COMMON_INTERCEPT_FUNCTION(vfscanf);
#else
#define INIT_SCANF
#endif

#if SANITIZER_INTERCEPT_ISOC99_SCANF
#define INIT_ISOC99_SCANF                      \
  COMMON_INTERCEPT_FUNCTION(__isoc99_scanf);   \
  COMMON_INTERCEPT_FUNCTION(__isoc99_sscanf);  \
  COMMON_INTERCEPT_FUNCTION(__isoc99_fscanf);  \
  COMMON_INTERCEPT_FUNCTION(__isoc99_vscanf);  \
  COMMON_INTERCEPT_FUNCTION(__isoc99_vsscanf); \
  COMMON_INTERCEPT_FUNCTION(__isoc99_vfscanf);
#else
#define INIT_ISOC99_SCANF
#endif

#if SANITIZER_INTERCEPT_PRINTF

#define VPRINTF_INTERCEPTOR_ENTER(vname, ...)                                  \
  void *ctx;                                                                   \
  COMMON_INTERCEPTOR_ENTER(ctx, vname, __VA_ARGS__);                           \
  va_list aq;                                                                  \
  va_copy(aq, ap);

#define VPRINTF_INTERCEPTOR_RETURN()                                           \
  va_end(aq);

#define VPRINTF_INTERCEPTOR_IMPL(vname, ...)                                   \
  {                                                                            \
    VPRINTF_INTERCEPTOR_ENTER(vname, __VA_ARGS__);                             \
    if (common_flags()->check_printf)                                          \
      printf_common(ctx, format, aq);                                          \
    int res = REAL(vname)(__VA_ARGS__);                                        \
    VPRINTF_INTERCEPTOR_RETURN();                                              \
    return res;                                                                \
  }

#define VSPRINTF_INTERCEPTOR_IMPL(vname, str, ...)                             \
  {                                                                            \
    VPRINTF_INTERCEPTOR_ENTER(vname, str, __VA_ARGS__)                         \
    if (common_flags()->check_printf) {                                        \
      printf_common(ctx, format, aq);                                          \
    }                                                                          \
    int res = REAL(vname)(str, __VA_ARGS__);                                   \
    if (res >= 0 && common_flags()->check_printf) {                            \
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, str, res + 1);                       \
    }                                                                          \
    VPRINTF_INTERCEPTOR_RETURN();                                              \
    return res;                                                                \
  }

#define VSNPRINTF_INTERCEPTOR_IMPL(vname, str, size, ...)                      \
  {                                                                            \
    VPRINTF_INTERCEPTOR_ENTER(vname, str, size, __VA_ARGS__)                   \
    if (common_flags()->check_printf) {                                        \
      printf_common(ctx, format, aq);                                          \
    }                                                                          \
    int res = REAL(vname)(str, size, __VA_ARGS__);                             \
    if (res >= 0 && common_flags()->check_printf) {                            \
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, str, Min(size, (SIZE_T)(res + 1)));  \
    }                                                                          \
    VPRINTF_INTERCEPTOR_RETURN();                                              \
    return res;                                                                \
  }

#define VASPRINTF_INTERCEPTOR_IMPL(vname, strp, ...)                           \
  {                                                                            \
    VPRINTF_INTERCEPTOR_ENTER(vname, strp, __VA_ARGS__)                        \
    if (common_flags()->check_printf) {                                        \
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, strp, sizeof(char *));               \
      printf_common(ctx, format, aq);                                          \
    }                                                                          \
    int res = REAL(vname)(strp, __VA_ARGS__);                                  \
    if (res >= 0 && common_flags()->check_printf) {                            \
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *strp, res + 1);                     \
    }                                                                          \
    VPRINTF_INTERCEPTOR_RETURN();                                              \
    return res;                                                                \
  }

INTERCEPTOR(int, vprintf, const char *format, va_list ap)
VPRINTF_INTERCEPTOR_IMPL(vprintf, format, ap)

INTERCEPTOR(int, vfprintf, void *stream, const char *format, va_list ap)
VPRINTF_INTERCEPTOR_IMPL(vfprintf, stream, format, ap)

INTERCEPTOR(int, vsnprintf, char *str, SIZE_T size, const char *format,
            va_list ap)
VSNPRINTF_INTERCEPTOR_IMPL(vsnprintf, str, size, format, ap)

INTERCEPTOR(int, vsprintf, char *str, const char *format, va_list ap)
VSPRINTF_INTERCEPTOR_IMPL(vsprintf, str, format, ap)

INTERCEPTOR(int, vasprintf, char **strp, const char *format, va_list ap)
VASPRINTF_INTERCEPTOR_IMPL(vasprintf, strp, format, ap)

#if SANITIZER_INTERCEPT_ISOC99_PRINTF
INTERCEPTOR(int, __isoc99_vprintf, const char *format, va_list ap)
VPRINTF_INTERCEPTOR_IMPL(__isoc99_vprintf, format, ap)

INTERCEPTOR(int, __isoc99_vfprintf, void *stream, const char *format,
            va_list ap)
VPRINTF_INTERCEPTOR_IMPL(__isoc99_vfprintf, stream, format, ap)

INTERCEPTOR(int, __isoc99_vsnprintf, char *str, SIZE_T size, const char *format,
            va_list ap)
VSNPRINTF_INTERCEPTOR_IMPL(__isoc99_vsnprintf, str, size, format, ap)

INTERCEPTOR(int, __isoc99_vsprintf, char *str, const char *format,
            va_list ap)
VSPRINTF_INTERCEPTOR_IMPL(__isoc99_vsprintf, str, format,
                          ap)

#endif  // SANITIZER_INTERCEPT_ISOC99_PRINTF

INTERCEPTOR(int, printf, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(printf, vprintf, format)

INTERCEPTOR(int, fprintf, void *stream, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(fprintf, vfprintf, stream, format)

INTERCEPTOR(int, sprintf, char *str, const char *format, ...) // NOLINT
FORMAT_INTERCEPTOR_IMPL(sprintf, vsprintf, str, format) // NOLINT

INTERCEPTOR(int, snprintf, char *str, SIZE_T size, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(snprintf, vsnprintf, str, size, format)

INTERCEPTOR(int, asprintf, char **strp, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(asprintf, vasprintf, strp, format)

#if SANITIZER_INTERCEPT_ISOC99_PRINTF
INTERCEPTOR(int, __isoc99_printf, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(__isoc99_printf, __isoc99_vprintf, format)

INTERCEPTOR(int, __isoc99_fprintf, void *stream, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(__isoc99_fprintf, __isoc99_vfprintf, stream, format)

INTERCEPTOR(int, __isoc99_sprintf, char *str, const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(__isoc99_sprintf, __isoc99_vsprintf, str, format)

INTERCEPTOR(int, __isoc99_snprintf, char *str, SIZE_T size,
            const char *format, ...)
FORMAT_INTERCEPTOR_IMPL(__isoc99_snprintf, __isoc99_vsnprintf, str, size,
                        format)

#endif  // SANITIZER_INTERCEPT_ISOC99_PRINTF

#endif  // SANITIZER_INTERCEPT_PRINTF

#if SANITIZER_INTERCEPT_PRINTF
#define INIT_PRINTF                     \
  COMMON_INTERCEPT_FUNCTION(printf);    \
  COMMON_INTERCEPT_FUNCTION(sprintf);   \
  COMMON_INTERCEPT_FUNCTION(snprintf);  \
  COMMON_INTERCEPT_FUNCTION(asprintf);  \
  COMMON_INTERCEPT_FUNCTION(fprintf);   \
  COMMON_INTERCEPT_FUNCTION(vprintf);   \
  COMMON_INTERCEPT_FUNCTION(vsprintf);  \
  COMMON_INTERCEPT_FUNCTION(vsnprintf); \
  COMMON_INTERCEPT_FUNCTION(vasprintf); \
  COMMON_INTERCEPT_FUNCTION(vfprintf);
#else
#define INIT_PRINTF
#endif

#if SANITIZER_INTERCEPT_ISOC99_PRINTF
#define INIT_ISOC99_PRINTF                       \
  COMMON_INTERCEPT_FUNCTION(__isoc99_printf);    \
  COMMON_INTERCEPT_FUNCTION(__isoc99_sprintf);   \
  COMMON_INTERCEPT_FUNCTION(__isoc99_snprintf);  \
  COMMON_INTERCEPT_FUNCTION(__isoc99_fprintf);   \
  COMMON_INTERCEPT_FUNCTION(__isoc99_vprintf);   \
  COMMON_INTERCEPT_FUNCTION(__isoc99_vsprintf);  \
  COMMON_INTERCEPT_FUNCTION(__isoc99_vsnprintf); \
  COMMON_INTERCEPT_FUNCTION(__isoc99_vfprintf);
#else
#define INIT_ISOC99_PRINTF
#endif

#if SANITIZER_INTERCEPT_IOCTL
#include "sanitizer_common_interceptors_ioctl.inc"
INTERCEPTOR(int, ioctl, int d, unsigned request, void *arg) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ioctl, d, request, arg);

  CHECK(ioctl_initialized);

  // Note: TSan does not use common flags, and they are zero-initialized.
  // This effectively disables ioctl handling in TSan.
  if (!common_flags()->handle_ioctl) return REAL(ioctl)(d, request, arg);

  const ioctl_desc *desc = ioctl_lookup(request);
  ioctl_desc decoded_desc;
  if (!desc) {
    VPrintf(2, "Decoding unknown ioctl 0x%x\n", request);
    if (!ioctl_decode(request, &decoded_desc))
      Printf("WARNING: failed decoding unknown ioctl 0x%x\n", request);
    else
      desc = &decoded_desc;
  }

  if (desc) ioctl_common_pre(ctx, desc, d, request, arg);
  int res = REAL(ioctl)(d, request, arg);
  // FIXME: some ioctls have different return values for success and failure.
  if (desc && res != -1) ioctl_common_post(ctx, desc, res, d, request, arg);
  return res;
}
#define INIT_IOCTL \
  ioctl_init();    \
  COMMON_INTERCEPT_FUNCTION(ioctl);
#else
#define INIT_IOCTL
#endif

#if SANITIZER_INTERCEPT_GETPWNAM_AND_FRIENDS
INTERCEPTOR(void *, getpwnam, const char *name) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpwnam, name);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  void *res = REAL(getpwnam)(name);
  if (res != 0) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_passwd_sz);
  return res;
}
INTERCEPTOR(void *, getpwuid, u32 uid) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpwuid, uid);
  void *res = REAL(getpwuid)(uid);
  if (res != 0) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_passwd_sz);
  return res;
}
INTERCEPTOR(void *, getgrnam, const char *name) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgrnam, name);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  void *res = REAL(getgrnam)(name);
  if (res != 0) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_group_sz);
  return res;
}
INTERCEPTOR(void *, getgrgid, u32 gid) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgrgid, gid);
  void *res = REAL(getgrgid)(gid);
  if (res != 0) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, struct_group_sz);
  return res;
}
#define INIT_GETPWNAM_AND_FRIENDS      \
  COMMON_INTERCEPT_FUNCTION(getpwnam); \
  COMMON_INTERCEPT_FUNCTION(getpwuid); \
  COMMON_INTERCEPT_FUNCTION(getgrnam); \
  COMMON_INTERCEPT_FUNCTION(getgrgid);
#else
#define INIT_GETPWNAM_AND_FRIENDS
#endif

#if SANITIZER_INTERCEPT_GETPWNAM_R_AND_FRIENDS
INTERCEPTOR(int, getpwnam_r, const char *name, void *pwd, char *buf,
            SIZE_T buflen, void **result) {
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
INTERCEPTOR(int, getpwuid_r, u32 uid, void *pwd, char *buf, SIZE_T buflen,
            void **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getpwuid_r, uid, pwd, buf, buflen, result);
  int res = REAL(getpwuid_r)(uid, pwd, buf, buflen, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, pwd, struct_passwd_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, buflen);
  }
  return res;
}
INTERCEPTOR(int, getgrnam_r, const char *name, void *grp, char *buf,
            SIZE_T buflen, void **result) {
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
INTERCEPTOR(int, getgrgid_r, u32 gid, void *grp, char *buf, SIZE_T buflen,
            void **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgrgid_r, gid, grp, buf, buflen, result);
  int res = REAL(getgrgid_r)(gid, grp, buf, buflen, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, grp, struct_group_sz);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, buflen);
  }
  return res;
}
#define INIT_GETPWNAM_R_AND_FRIENDS      \
  COMMON_INTERCEPT_FUNCTION(getpwnam_r); \
  COMMON_INTERCEPT_FUNCTION(getpwuid_r); \
  COMMON_INTERCEPT_FUNCTION(getgrnam_r); \
  COMMON_INTERCEPT_FUNCTION(getgrgid_r);
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
#define INIT_CLOCK_GETTIME                  \
  COMMON_INTERCEPT_FUNCTION(clock_getres);  \
  COMMON_INTERCEPT_FUNCTION(clock_gettime); \
  COMMON_INTERCEPT_FUNCTION(clock_settime);
#else
#define INIT_CLOCK_GETTIME
#endif

#if SANITIZER_INTERCEPT_GETITIMER
INTERCEPTOR(int, getitimer, int which, void *curr_value) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getitimer, which, curr_value);
  int res = REAL(getitimer)(which, curr_value);
  if (!res && curr_value) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, curr_value, struct_itimerval_sz);
  }
  return res;
}
INTERCEPTOR(int, setitimer, int which, const void *new_value, void *old_value) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, setitimer, which, new_value, old_value);
  if (new_value)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, new_value, struct_itimerval_sz);
  int res = REAL(setitimer)(which, new_value, old_value);
  if (!res && old_value) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, old_value, struct_itimerval_sz);
  }
  return res;
}
#define INIT_GETITIMER                  \
  COMMON_INTERCEPT_FUNCTION(getitimer); \
  COMMON_INTERCEPT_FUNCTION(setitimer);
#else
#define INIT_GETITIMER
#endif

#if SANITIZER_INTERCEPT_GLOB
static void unpoison_glob_t(void *ctx, __sanitizer_glob_t *pglob) {
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, pglob, sizeof(*pglob));
  // +1 for NULL pointer at the end.
  if (pglob->gl_pathv)
    COMMON_INTERCEPTOR_WRITE_RANGE(
        ctx, pglob->gl_pathv, (pglob->gl_pathc + 1) * sizeof(*pglob->gl_pathv));
  for (SIZE_T i = 0; i < pglob->gl_pathc; ++i) {
    char *p = pglob->gl_pathv[i];
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p, REAL(strlen)(p) + 1);
  }
}

static THREADLOCAL __sanitizer_glob_t *pglob_copy;
static THREADLOCAL void *glob_ctx;

static void wrapped_gl_closedir(void *dir) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(glob_ctx, 1);
  IndirectExternCall(pglob_copy->gl_closedir)(dir);
}

static void *wrapped_gl_readdir(void *dir) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(glob_ctx, 1);
  return IndirectExternCall(pglob_copy->gl_readdir)(dir);
}

static void *wrapped_gl_opendir(const char *s) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(glob_ctx, 1);
  COMMON_INTERCEPTOR_WRITE_RANGE(glob_ctx, s, REAL(strlen)(s) + 1);
  return IndirectExternCall(pglob_copy->gl_opendir)(s);
}

static int wrapped_gl_lstat(const char *s, void *st) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(glob_ctx, 2);
  COMMON_INTERCEPTOR_WRITE_RANGE(glob_ctx, s, REAL(strlen)(s) + 1);
  return IndirectExternCall(pglob_copy->gl_lstat)(s, st);
}

static int wrapped_gl_stat(const char *s, void *st) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(glob_ctx, 2);
  COMMON_INTERCEPTOR_WRITE_RANGE(glob_ctx, s, REAL(strlen)(s) + 1);
  return IndirectExternCall(pglob_copy->gl_stat)(s, st);
}

INTERCEPTOR(int, glob, const char *pattern, int flags,
            int (*errfunc)(const char *epath, int eerrno),
            __sanitizer_glob_t *pglob) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, glob, pattern, flags, errfunc, pglob);
  __sanitizer_glob_t glob_copy = {
      0,                  0,                   0,
      0,                  wrapped_gl_closedir, wrapped_gl_readdir,
      wrapped_gl_opendir, wrapped_gl_lstat,    wrapped_gl_stat};
  if (flags & glob_altdirfunc) {
    Swap(pglob->gl_closedir, glob_copy.gl_closedir);
    Swap(pglob->gl_readdir, glob_copy.gl_readdir);
    Swap(pglob->gl_opendir, glob_copy.gl_opendir);
    Swap(pglob->gl_lstat, glob_copy.gl_lstat);
    Swap(pglob->gl_stat, glob_copy.gl_stat);
    pglob_copy = &glob_copy;
    glob_ctx = ctx;
  }
  int res = REAL(glob)(pattern, flags, errfunc, pglob);
  if (flags & glob_altdirfunc) {
    Swap(pglob->gl_closedir, glob_copy.gl_closedir);
    Swap(pglob->gl_readdir, glob_copy.gl_readdir);
    Swap(pglob->gl_opendir, glob_copy.gl_opendir);
    Swap(pglob->gl_lstat, glob_copy.gl_lstat);
    Swap(pglob->gl_stat, glob_copy.gl_stat);
  }
  pglob_copy = 0;
  glob_ctx = 0;
  if ((!res || res == glob_nomatch) && pglob) unpoison_glob_t(ctx, pglob);
  return res;
}

INTERCEPTOR(int, glob64, const char *pattern, int flags,
            int (*errfunc)(const char *epath, int eerrno),
            __sanitizer_glob_t *pglob) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, glob64, pattern, flags, errfunc, pglob);
  __sanitizer_glob_t glob_copy = {
      0,                  0,                   0,
      0,                  wrapped_gl_closedir, wrapped_gl_readdir,
      wrapped_gl_opendir, wrapped_gl_lstat,    wrapped_gl_stat};
  if (flags & glob_altdirfunc) {
    Swap(pglob->gl_closedir, glob_copy.gl_closedir);
    Swap(pglob->gl_readdir, glob_copy.gl_readdir);
    Swap(pglob->gl_opendir, glob_copy.gl_opendir);
    Swap(pglob->gl_lstat, glob_copy.gl_lstat);
    Swap(pglob->gl_stat, glob_copy.gl_stat);
    pglob_copy = &glob_copy;
    glob_ctx = ctx;
  }
  int res = REAL(glob64)(pattern, flags, errfunc, pglob);
  if (flags & glob_altdirfunc) {
    Swap(pglob->gl_closedir, glob_copy.gl_closedir);
    Swap(pglob->gl_readdir, glob_copy.gl_readdir);
    Swap(pglob->gl_opendir, glob_copy.gl_opendir);
    Swap(pglob->gl_lstat, glob_copy.gl_lstat);
    Swap(pglob->gl_stat, glob_copy.gl_stat);
  }
  pglob_copy = 0;
  glob_ctx = 0;
  if ((!res || res == glob_nomatch) && pglob) unpoison_glob_t(ctx, pglob);
  return res;
}
#define INIT_GLOB                  \
  COMMON_INTERCEPT_FUNCTION(glob); \
  COMMON_INTERCEPT_FUNCTION(glob64);
#else  // SANITIZER_INTERCEPT_GLOB
#define INIT_GLOB
#endif  // SANITIZER_INTERCEPT_GLOB

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
    if (status) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, status, sizeof(*status));
    if (rusage) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, rusage, struct_rusage_sz);
  }
  return res;
}
#if SANITIZER_ANDROID
INTERCEPTOR(int, __wait4, int pid, int *status, int options, void *rusage) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, __wait4, pid, status, options, rusage);
  int res = REAL(__wait4)(pid, status, options, rusage);
  if (res != -1) {
    if (status) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, status, sizeof(*status));
    if (rusage) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, rusage, struct_rusage_sz);
  }
  return res;
}
#define INIT_WAIT4 COMMON_INTERCEPT_FUNCTION(__wait4);
#else
INTERCEPTOR(int, wait4, int pid, int *status, int options, void *rusage) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wait4, pid, status, options, rusage);
  int res = REAL(wait4)(pid, status, options, rusage);
  if (res != -1) {
    if (status) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, status, sizeof(*status));
    if (rusage) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, rusage, struct_rusage_sz);
  }
  return res;
}
#define INIT_WAIT4 COMMON_INTERCEPT_FUNCTION(wait4);
#endif  // SANITIZER_ANDROID
#define INIT_WAIT                     \
  COMMON_INTERCEPT_FUNCTION(wait);    \
  COMMON_INTERCEPT_FUNCTION(waitid);  \
  COMMON_INTERCEPT_FUNCTION(waitpid); \
  COMMON_INTERCEPT_FUNCTION(wait3);
#else
#define INIT_WAIT
#define INIT_WAIT4
#endif

#if SANITIZER_INTERCEPT_INET
INTERCEPTOR(char *, inet_ntop, int af, const void *src, char *dst, u32 size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, inet_ntop, af, src, dst, size);
  uptr sz = __sanitizer_in_addr_sz(af);
  if (sz) COMMON_INTERCEPTOR_READ_RANGE(ctx, src, sz);
  // FIXME: figure out read size based on the address family.
  char *res = REAL(inet_ntop)(af, src, dst, size);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
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
#define INIT_INET                       \
  COMMON_INTERCEPT_FUNCTION(inet_ntop); \
  COMMON_INTERCEPT_FUNCTION(inet_pton);
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
#define INIT_INET_ATON COMMON_INTERCEPT_FUNCTION(inet_aton);
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
#define INIT_PTHREAD_GETSCHEDPARAM \
  COMMON_INTERCEPT_FUNCTION(pthread_getschedparam);
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
#define INIT_GETADDRINFO COMMON_INTERCEPT_FUNCTION(getaddrinfo);
#else
#define INIT_GETADDRINFO
#endif

#if SANITIZER_INTERCEPT_GETNAMEINFO
INTERCEPTOR(int, getnameinfo, void *sockaddr, unsigned salen, char *host,
            unsigned hostlen, char *serv, unsigned servlen, int flags) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getnameinfo, sockaddr, salen, host, hostlen,
                           serv, servlen, flags);
  // FIXME: consider adding READ_RANGE(sockaddr, salen)
  // There is padding in in_addr that may make this too noisy
  int res =
      REAL(getnameinfo)(sockaddr, salen, host, hostlen, serv, servlen, flags);
  if (res == 0) {
    if (host && hostlen)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, host, REAL(strlen)(host) + 1);
    if (serv && servlen)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, serv, REAL(strlen)(serv) + 1);
  }
  return res;
}
#define INIT_GETNAMEINFO COMMON_INTERCEPT_FUNCTION(getnameinfo);
#else
#define INIT_GETNAMEINFO
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
#define INIT_GETSOCKNAME COMMON_INTERCEPT_FUNCTION(getsockname);
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

INTERCEPTOR(struct __sanitizer_hostent *, gethostent, int fake) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostent, fake);
  struct __sanitizer_hostent *res = REAL(gethostent)(fake);
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
#define INIT_GETHOSTBYNAME                  \
  COMMON_INTERCEPT_FUNCTION(gethostent);    \
  COMMON_INTERCEPT_FUNCTION(gethostbyaddr); \
  COMMON_INTERCEPT_FUNCTION(gethostbyname); \
  COMMON_INTERCEPT_FUNCTION(gethostbyname2);
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
  if (result) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (res == 0 && *result) write_hostent(ctx, *result);
  }
  if (h_errnop)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
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
  if (result) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (res == 0 && *result) write_hostent(ctx, *result);
  }
  if (h_errnop)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
  return res;
}

INTERCEPTOR(int, gethostbyname_r, char *name, struct __sanitizer_hostent *ret,
            char *buf, SIZE_T buflen, __sanitizer_hostent **result,
            int *h_errnop) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, gethostbyname_r, name, ret, buf, buflen, result,
                           h_errnop);
  int res = REAL(gethostbyname_r)(name, ret, buf, buflen, result, h_errnop);
  if (result) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (res == 0 && *result) write_hostent(ctx, *result);
  }
  if (h_errnop)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
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
  if (result) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (res == 0 && *result) write_hostent(ctx, *result);
  }
  if (h_errnop)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, h_errnop, sizeof(*h_errnop));
  return res;
}
#define INIT_GETHOSTBYNAME_R                  \
  COMMON_INTERCEPT_FUNCTION(gethostent_r);    \
  COMMON_INTERCEPT_FUNCTION(gethostbyaddr_r); \
  COMMON_INTERCEPT_FUNCTION(gethostbyname_r); \
  COMMON_INTERCEPT_FUNCTION(gethostbyname2_r);
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
#define INIT_GETSOCKOPT COMMON_INTERCEPT_FUNCTION(getsockopt);
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
    if (fd >= 0) COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, fd2);
    if (addr && addrlen)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, Min(*addrlen, addrlen0));
  }
  return fd2;
}
#define INIT_ACCEPT COMMON_INTERCEPT_FUNCTION(accept);
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
    if (fd >= 0) COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, fd2);
    if (addr && addrlen)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, Min(*addrlen, addrlen0));
  }
  return fd2;
}
#define INIT_ACCEPT4 COMMON_INTERCEPT_FUNCTION(accept4);
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
#define INIT_MODF                   \
  COMMON_INTERCEPT_FUNCTION(modf);  \
  COMMON_INTERCEPT_FUNCTION(modff); \
  COMMON_INTERCEPT_FUNCTION(modfl);
#else
#define INIT_MODF
#endif

#if SANITIZER_INTERCEPT_RECVMSG
static void write_msghdr(void *ctx, struct __sanitizer_msghdr *msg,
                         SSIZE_T maxlen) {
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg, sizeof(*msg));
  if (msg->msg_name && msg->msg_namelen)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg->msg_name, msg->msg_namelen);
  if (msg->msg_iov && msg->msg_iovlen)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg->msg_iov,
                                   sizeof(*msg->msg_iov) * msg->msg_iovlen);
  write_iovec(ctx, msg->msg_iov, msg->msg_iovlen, maxlen);
  if (msg->msg_control && msg->msg_controllen)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, msg->msg_control, msg->msg_controllen);
}

INTERCEPTOR(SSIZE_T, recvmsg, int fd, struct __sanitizer_msghdr *msg,
            int flags) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, recvmsg, fd, msg, flags);
  SSIZE_T res = REAL(recvmsg)(fd, msg, flags);
  if (res >= 0) {
    if (fd >= 0) COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd);
    if (msg) {
      write_msghdr(ctx, msg, res);
      COMMON_INTERCEPTOR_HANDLE_RECVMSG(ctx, msg);
    }
  }
  return res;
}
#define INIT_RECVMSG COMMON_INTERCEPT_FUNCTION(recvmsg);
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
#define INIT_GETPEERNAME COMMON_INTERCEPT_FUNCTION(getpeername);
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
#define INIT_SYSINFO COMMON_INTERCEPT_FUNCTION(sysinfo);
#else
#define INIT_SYSINFO
#endif

#if SANITIZER_INTERCEPT_READDIR
INTERCEPTOR(__sanitizer_dirent *, readdir, void *dirp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir, dirp);
  __sanitizer_dirent *res = REAL(readdir)(dirp);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, res->d_reclen);
  return res;
}

INTERCEPTOR(int, readdir_r, void *dirp, __sanitizer_dirent *entry,
            __sanitizer_dirent **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir_r, dirp, entry, result);
  int res = REAL(readdir_r)(dirp, entry, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (*result)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *result, (*result)->d_reclen);
  }
  return res;
}

#define INIT_READDIR                  \
  COMMON_INTERCEPT_FUNCTION(readdir); \
  COMMON_INTERCEPT_FUNCTION(readdir_r);
#else
#define INIT_READDIR
#endif

#if SANITIZER_INTERCEPT_READDIR64
INTERCEPTOR(__sanitizer_dirent64 *, readdir64, void *dirp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir64, dirp);
  __sanitizer_dirent64 *res = REAL(readdir64)(dirp);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, res->d_reclen);
  return res;
}

INTERCEPTOR(int, readdir64_r, void *dirp, __sanitizer_dirent64 *entry,
            __sanitizer_dirent64 **result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, readdir64_r, dirp, entry, result);
  int res = REAL(readdir64_r)(dirp, entry, result);
  if (!res) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
    if (*result)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *result, (*result)->d_reclen);
  }
  return res;
}
#define INIT_READDIR64                  \
  COMMON_INTERCEPT_FUNCTION(readdir64); \
  COMMON_INTERCEPT_FUNCTION(readdir64_r);
#else
#define INIT_READDIR64
#endif

#if SANITIZER_INTERCEPT_PTRACE
INTERCEPTOR(uptr, ptrace, int request, int pid, void *addr, void *data) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ptrace, request, pid, addr, data);

  if (data) {
    if (request == ptrace_setregs)
      COMMON_INTERCEPTOR_READ_RANGE(ctx, data, struct_user_regs_struct_sz);
    else if (request == ptrace_setfpregs)
      COMMON_INTERCEPTOR_READ_RANGE(ctx, data, struct_user_fpregs_struct_sz);
    else if (request == ptrace_setfpxregs)
      COMMON_INTERCEPTOR_READ_RANGE(ctx, data, struct_user_fpxregs_struct_sz);
    else if (request == ptrace_setsiginfo)
      COMMON_INTERCEPTOR_READ_RANGE(ctx, data, siginfo_t_sz);
    else if (request == ptrace_setregset) {
      __sanitizer_iovec *iov = (__sanitizer_iovec *)data;
      COMMON_INTERCEPTOR_READ_RANGE(ctx, iov->iov_base, iov->iov_len);
    }
  }

  uptr res = REAL(ptrace)(request, pid, addr, data);

  if (!res && data) {
    // Note that PEEK* requests assing different meaning to the return value.
    // This function does not handle them (nor does it need to).
    if (request == ptrace_getregs)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, data, struct_user_regs_struct_sz);
    else if (request == ptrace_getfpregs)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, data, struct_user_fpregs_struct_sz);
    else if (request == ptrace_getfpxregs)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, data, struct_user_fpxregs_struct_sz);
    else if (request == ptrace_getsiginfo)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, data, siginfo_t_sz);
    else if (request == ptrace_getregset) {
      __sanitizer_iovec *iov = (__sanitizer_iovec *)data;
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, iov->iov_base, iov->iov_len);
    }
  }
  return res;
}

#define INIT_PTRACE COMMON_INTERCEPT_FUNCTION(ptrace);
#else
#define INIT_PTRACE
#endif

#if SANITIZER_INTERCEPT_SETLOCALE
INTERCEPTOR(char *, setlocale, int category, char *locale) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, setlocale, category, locale);
  if (locale)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, locale, REAL(strlen)(locale) + 1);
  char *res = REAL(setlocale)(category, locale);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}

#define INIT_SETLOCALE COMMON_INTERCEPT_FUNCTION(setlocale);
#else
#define INIT_SETLOCALE
#endif

#if SANITIZER_INTERCEPT_GETCWD
INTERCEPTOR(char *, getcwd, char *buf, SIZE_T size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getcwd, buf, size);
  char *res = REAL(getcwd)(buf, size);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
#define INIT_GETCWD COMMON_INTERCEPT_FUNCTION(getcwd);
#else
#define INIT_GETCWD
#endif

#if SANITIZER_INTERCEPT_GET_CURRENT_DIR_NAME
INTERCEPTOR(char *, get_current_dir_name, int fake) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, get_current_dir_name, fake);
  char *res = REAL(get_current_dir_name)(fake);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}

#define INIT_GET_CURRENT_DIR_NAME \
  COMMON_INTERCEPT_FUNCTION(get_current_dir_name);
#else
#define INIT_GET_CURRENT_DIR_NAME
#endif

#if SANITIZER_INTERCEPT_STRTOIMAX
INTERCEPTOR(INTMAX_T, strtoimax, const char *nptr, char **endptr, int base) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strtoimax, nptr, endptr, base);
  INTMAX_T res = REAL(strtoimax)(nptr, endptr, base);
  if (endptr) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, endptr, sizeof(*endptr));
  return res;
}

INTERCEPTOR(INTMAX_T, strtoumax, const char *nptr, char **endptr, int base) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strtoumax, nptr, endptr, base);
  INTMAX_T res = REAL(strtoumax)(nptr, endptr, base);
  if (endptr) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, endptr, sizeof(*endptr));
  return res;
}

#define INIT_STRTOIMAX                  \
  COMMON_INTERCEPT_FUNCTION(strtoimax); \
  COMMON_INTERCEPT_FUNCTION(strtoumax);
#else
#define INIT_STRTOIMAX
#endif

#if SANITIZER_INTERCEPT_MBSTOWCS
INTERCEPTOR(SIZE_T, mbstowcs, wchar_t *dest, const char *src, SIZE_T len) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, mbstowcs, dest, src, len);
  SIZE_T res = REAL(mbstowcs)(dest, src, len);
  if (res != (SIZE_T) - 1 && dest) {
    SIZE_T write_cnt = res + (res < len);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dest, write_cnt * sizeof(wchar_t));
  }
  return res;
}

INTERCEPTOR(SIZE_T, mbsrtowcs, wchar_t *dest, const char **src, SIZE_T len,
            void *ps) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, mbsrtowcs, dest, src, len, ps);
  if (src) COMMON_INTERCEPTOR_READ_RANGE(ctx, src, sizeof(*src));
  if (ps) COMMON_INTERCEPTOR_READ_RANGE(ctx, ps, mbstate_t_sz);
  SIZE_T res = REAL(mbsrtowcs)(dest, src, len, ps);
  if (res != (SIZE_T)(-1) && dest && src) {
    // This function, and several others, may or may not write the terminating
    // \0 character. They write it iff they clear *src.
    SIZE_T write_cnt = res + !*src;
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dest, write_cnt * sizeof(wchar_t));
  }
  return res;
}

#define INIT_MBSTOWCS                  \
  COMMON_INTERCEPT_FUNCTION(mbstowcs); \
  COMMON_INTERCEPT_FUNCTION(mbsrtowcs);
#else
#define INIT_MBSTOWCS
#endif

#if SANITIZER_INTERCEPT_MBSNRTOWCS
INTERCEPTOR(SIZE_T, mbsnrtowcs, wchar_t *dest, const char **src, SIZE_T nms,
            SIZE_T len, void *ps) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, mbsnrtowcs, dest, src, nms, len, ps);
  if (src) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, src, sizeof(*src));
    if (nms) COMMON_INTERCEPTOR_READ_RANGE(ctx, *src, nms);
  }
  if (ps) COMMON_INTERCEPTOR_READ_RANGE(ctx, ps, mbstate_t_sz);
  SIZE_T res = REAL(mbsnrtowcs)(dest, src, nms, len, ps);
  if (res != (SIZE_T)(-1) && dest && src) {
    SIZE_T write_cnt = res + !*src;
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dest, write_cnt * sizeof(wchar_t));
  }
  return res;
}

#define INIT_MBSNRTOWCS COMMON_INTERCEPT_FUNCTION(mbsnrtowcs);
#else
#define INIT_MBSNRTOWCS
#endif

#if SANITIZER_INTERCEPT_WCSTOMBS
INTERCEPTOR(SIZE_T, wcstombs, char *dest, const wchar_t *src, SIZE_T len) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wcstombs, dest, src, len);
  SIZE_T res = REAL(wcstombs)(dest, src, len);
  if (res != (SIZE_T) - 1 && dest) {
    SIZE_T write_cnt = res + (res < len);
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dest, write_cnt);
  }
  return res;
}

INTERCEPTOR(SIZE_T, wcsrtombs, char *dest, const wchar_t **src, SIZE_T len,
            void *ps) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wcsrtombs, dest, src, len, ps);
  if (src) COMMON_INTERCEPTOR_READ_RANGE(ctx, src, sizeof(*src));
  if (ps) COMMON_INTERCEPTOR_READ_RANGE(ctx, ps, mbstate_t_sz);
  SIZE_T res = REAL(wcsrtombs)(dest, src, len, ps);
  if (res != (SIZE_T) - 1 && dest && src) {
    SIZE_T write_cnt = res + !*src;
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dest, write_cnt);
  }
  return res;
}

#define INIT_WCSTOMBS                  \
  COMMON_INTERCEPT_FUNCTION(wcstombs); \
  COMMON_INTERCEPT_FUNCTION(wcsrtombs);
#else
#define INIT_WCSTOMBS
#endif

#if SANITIZER_INTERCEPT_WCSNRTOMBS
INTERCEPTOR(SIZE_T, wcsnrtombs, char *dest, const wchar_t **src, SIZE_T nms,
            SIZE_T len, void *ps) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wcsnrtombs, dest, src, nms, len, ps);
  if (src) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, src, sizeof(*src));
    if (nms) COMMON_INTERCEPTOR_READ_RANGE(ctx, *src, nms);
  }
  if (ps) COMMON_INTERCEPTOR_READ_RANGE(ctx, ps, mbstate_t_sz);
  SIZE_T res = REAL(wcsnrtombs)(dest, src, nms, len, ps);
  if (res != (SIZE_T) - 1 && dest && src) {
    SIZE_T write_cnt = res + !*src;
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dest, write_cnt);
  }
  return res;
}

#define INIT_WCSNRTOMBS COMMON_INTERCEPT_FUNCTION(wcsnrtombs);
#else
#define INIT_WCSNRTOMBS
#endif

#if SANITIZER_INTERCEPT_TCGETATTR
INTERCEPTOR(int, tcgetattr, int fd, void *termios_p) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, tcgetattr, fd, termios_p);
  int res = REAL(tcgetattr)(fd, termios_p);
  if (!res && termios_p)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, termios_p, struct_termios_sz);
  return res;
}

#define INIT_TCGETATTR COMMON_INTERCEPT_FUNCTION(tcgetattr);
#else
#define INIT_TCGETATTR
#endif

#if SANITIZER_INTERCEPT_REALPATH
INTERCEPTOR(char *, realpath, const char *path, char *resolved_path) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, realpath, path, resolved_path);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);

  // Workaround a bug in glibc where dlsym(RTLD_NEXT, ...) returns the oldest
  // version of a versioned symbol. For realpath(), this gives us something
  // (called __old_realpath) that does not handle NULL in the second argument.
  // Handle it as part of the interceptor.
  char *allocated_path = 0;
  if (!resolved_path)
    allocated_path = resolved_path = (char *)WRAP(malloc)(path_max + 1);

  char *res = REAL(realpath)(path, resolved_path);
  if (allocated_path && !res) WRAP(free)(allocated_path);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
#define INIT_REALPATH COMMON_INTERCEPT_FUNCTION(realpath);
#else
#define INIT_REALPATH
#endif

#if SANITIZER_INTERCEPT_CANONICALIZE_FILE_NAME
INTERCEPTOR(char *, canonicalize_file_name, const char *path) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, canonicalize_file_name, path);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  char *res = REAL(canonicalize_file_name)(path);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
#define INIT_CANONICALIZE_FILE_NAME \
  COMMON_INTERCEPT_FUNCTION(canonicalize_file_name);
#else
#define INIT_CANONICALIZE_FILE_NAME
#endif

#if SANITIZER_INTERCEPT_CONFSTR
INTERCEPTOR(SIZE_T, confstr, int name, char *buf, SIZE_T len) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, confstr, name, buf, len);
  SIZE_T res = REAL(confstr)(name, buf, len);
  if (buf && res)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, res < len ? res : len);
  return res;
}
#define INIT_CONFSTR COMMON_INTERCEPT_FUNCTION(confstr);
#else
#define INIT_CONFSTR
#endif

#if SANITIZER_INTERCEPT_SCHED_GETAFFINITY
INTERCEPTOR(int, sched_getaffinity, int pid, SIZE_T cpusetsize, void *mask) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sched_getaffinity, pid, cpusetsize, mask);
  int res = REAL(sched_getaffinity)(pid, cpusetsize, mask);
  if (mask && !res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, mask, cpusetsize);
  return res;
}
#define INIT_SCHED_GETAFFINITY COMMON_INTERCEPT_FUNCTION(sched_getaffinity);
#else
#define INIT_SCHED_GETAFFINITY
#endif

#if SANITIZER_INTERCEPT_STRERROR
INTERCEPTOR(char *, strerror, int errnum) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strerror, errnum);
  char *res = REAL(strerror)(errnum);
  if (res) COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
#define INIT_STRERROR COMMON_INTERCEPT_FUNCTION(strerror);
#else
#define INIT_STRERROR
#endif

#if SANITIZER_INTERCEPT_STRERROR_R
INTERCEPTOR(char *, strerror_r, int errnum, char *buf, SIZE_T buflen) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strerror_r, errnum, buf, buflen);
  char *res = REAL(strerror_r)(errnum, buf, buflen);
  // There are 2 versions of strerror_r:
  //  * POSIX version returns 0 on success, negative error code on failure,
  //    writes message to buf.
  //  * GNU version returns message pointer, which points to either buf or some
  //    static storage.
  SIZE_T posix_res = (SIZE_T)res;
  if (posix_res < 1024 || posix_res > (SIZE_T) - 1024) {
    // POSIX version. Spec is not clear on whether buf is NULL-terminated.
    // At least on OSX, buf contents are valid even when the call fails.
    SIZE_T sz = internal_strnlen(buf, buflen);
    if (sz < buflen) ++sz;
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, sz);
  } else {
    // GNU version.
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
#define INIT_STRERROR_R COMMON_INTERCEPT_FUNCTION(strerror_r);
#else
#define INIT_STRERROR_R
#endif

#if SANITIZER_INTERCEPT_XPG_STRERROR_R
INTERCEPTOR(int, __xpg_strerror_r, int errnum, char *buf, SIZE_T buflen) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, __xpg_strerror_r, errnum, buf, buflen);
  int res = REAL(__xpg_strerror_r)(errnum, buf, buflen);
  // This version always returns a null-terminated string.
  if (buf && buflen)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, REAL(strlen)(buf) + 1);
  return res;
}
#define INIT_XPG_STRERROR_R COMMON_INTERCEPT_FUNCTION(__xpg_strerror_r);
#else
#define INIT_XPG_STRERROR_R
#endif

#if SANITIZER_INTERCEPT_SCANDIR
typedef int (*scandir_filter_f)(const struct __sanitizer_dirent *);
typedef int (*scandir_compar_f)(const struct __sanitizer_dirent **,
                                const struct __sanitizer_dirent **);

static THREADLOCAL void *scandir_ctx;
static THREADLOCAL scandir_filter_f scandir_filter;
static THREADLOCAL scandir_compar_f scandir_compar;

static int wrapped_scandir_filter(const struct __sanitizer_dirent *dir) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(scandir_ctx, 1);
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir_ctx, dir, dir->d_reclen);
  return IndirectExternCall(scandir_filter)(dir);
}

static int wrapped_scandir_compar(const struct __sanitizer_dirent **a,
                                  const struct __sanitizer_dirent **b) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(scandir_ctx, 2);
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir_ctx, a, sizeof(*a));
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir_ctx, *a, (*a)->d_reclen);
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir_ctx, b, sizeof(*b));
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir_ctx, *b, (*b)->d_reclen);
  return IndirectExternCall(scandir_compar)(a, b);
}

INTERCEPTOR(int, scandir, char *dirp, __sanitizer_dirent ***namelist,
            scandir_filter_f filter, scandir_compar_f compar) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, scandir, dirp, namelist, filter, compar);
  if (dirp) COMMON_INTERCEPTOR_READ_RANGE(ctx, dirp, REAL(strlen)(dirp) + 1);
  CHECK_EQ(0, scandir_ctx);
  scandir_ctx = ctx;
  scandir_filter = filter;
  scandir_compar = compar;
  int res = REAL(scandir)(dirp, namelist, filter ? wrapped_scandir_filter : 0,
                          compar ? wrapped_scandir_compar : 0);
  scandir_ctx = 0;
  scandir_filter = 0;
  scandir_compar = 0;
  if (namelist && res > 0) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, namelist, sizeof(*namelist));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *namelist, sizeof(**namelist) * res);
    for (int i = 0; i < res; ++i)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, (*namelist)[i],
                                     (*namelist)[i]->d_reclen);
  }
  return res;
}
#define INIT_SCANDIR COMMON_INTERCEPT_FUNCTION(scandir);
#else
#define INIT_SCANDIR
#endif

#if SANITIZER_INTERCEPT_SCANDIR64
typedef int (*scandir64_filter_f)(const struct __sanitizer_dirent64 *);
typedef int (*scandir64_compar_f)(const struct __sanitizer_dirent64 **,
                                  const struct __sanitizer_dirent64 **);

static THREADLOCAL void *scandir64_ctx;
static THREADLOCAL scandir64_filter_f scandir64_filter;
static THREADLOCAL scandir64_compar_f scandir64_compar;

static int wrapped_scandir64_filter(const struct __sanitizer_dirent64 *dir) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(scandir64_ctx, 1);
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir64_ctx, dir, dir->d_reclen);
  return IndirectExternCall(scandir64_filter)(dir);
}

static int wrapped_scandir64_compar(const struct __sanitizer_dirent64 **a,
                                    const struct __sanitizer_dirent64 **b) {
  COMMON_INTERCEPTOR_UNPOISON_PARAM(scandir64_ctx, 2);
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir64_ctx, a, sizeof(*a));
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir64_ctx, *a, (*a)->d_reclen);
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir64_ctx, b, sizeof(*b));
  COMMON_INTERCEPTOR_WRITE_RANGE(scandir64_ctx, *b, (*b)->d_reclen);
  return IndirectExternCall(scandir64_compar)(a, b);
}

INTERCEPTOR(int, scandir64, char *dirp, __sanitizer_dirent64 ***namelist,
            scandir64_filter_f filter, scandir64_compar_f compar) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, scandir64, dirp, namelist, filter, compar);
  if (dirp) COMMON_INTERCEPTOR_READ_RANGE(ctx, dirp, REAL(strlen)(dirp) + 1);
  CHECK_EQ(0, scandir64_ctx);
  scandir64_ctx = ctx;
  scandir64_filter = filter;
  scandir64_compar = compar;
  int res =
      REAL(scandir64)(dirp, namelist, filter ? wrapped_scandir64_filter : 0,
                      compar ? wrapped_scandir64_compar : 0);
  scandir64_ctx = 0;
  scandir64_filter = 0;
  scandir64_compar = 0;
  if (namelist && res > 0) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, namelist, sizeof(*namelist));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *namelist, sizeof(**namelist) * res);
    for (int i = 0; i < res; ++i)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, (*namelist)[i],
                                     (*namelist)[i]->d_reclen);
  }
  return res;
}
#define INIT_SCANDIR64 COMMON_INTERCEPT_FUNCTION(scandir64);
#else
#define INIT_SCANDIR64
#endif

#if SANITIZER_INTERCEPT_GETGROUPS
INTERCEPTOR(int, getgroups, int size, u32 *lst) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getgroups, size, lst);
  int res = REAL(getgroups)(size, lst);
  if (res && lst) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, lst, res * sizeof(*lst));
  return res;
}
#define INIT_GETGROUPS COMMON_INTERCEPT_FUNCTION(getgroups);
#else
#define INIT_GETGROUPS
#endif

#if SANITIZER_INTERCEPT_POLL
static void read_pollfd(void *ctx, __sanitizer_pollfd *fds,
                        __sanitizer_nfds_t nfds) {
  for (unsigned i = 0; i < nfds; ++i) {
    COMMON_INTERCEPTOR_READ_RANGE(ctx, &fds[i].fd, sizeof(fds[i].fd));
    COMMON_INTERCEPTOR_READ_RANGE(ctx, &fds[i].events, sizeof(fds[i].events));
  }
}

static void write_pollfd(void *ctx, __sanitizer_pollfd *fds,
                         __sanitizer_nfds_t nfds) {
  for (unsigned i = 0; i < nfds; ++i)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, &fds[i].revents,
                                   sizeof(fds[i].revents));
}

INTERCEPTOR(int, poll, __sanitizer_pollfd *fds, __sanitizer_nfds_t nfds,
            int timeout) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, poll, fds, nfds, timeout);
  if (fds && nfds) read_pollfd(ctx, fds, nfds);
  int res = COMMON_INTERCEPTOR_BLOCK_REAL(poll)(fds, nfds, timeout);
  if (fds && nfds) write_pollfd(ctx, fds, nfds);
  return res;
}
#define INIT_POLL COMMON_INTERCEPT_FUNCTION(poll);
#else
#define INIT_POLL
#endif

#if SANITIZER_INTERCEPT_PPOLL
INTERCEPTOR(int, ppoll, __sanitizer_pollfd *fds, __sanitizer_nfds_t nfds,
            void *timeout_ts, __sanitizer_sigset_t *sigmask) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ppoll, fds, nfds, timeout_ts, sigmask);
  if (fds && nfds) read_pollfd(ctx, fds, nfds);
  if (timeout_ts)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, timeout_ts, struct_timespec_sz);
  // FIXME: read sigmask when all of sigemptyset, etc are intercepted.
  int res =
      COMMON_INTERCEPTOR_BLOCK_REAL(ppoll)(fds, nfds, timeout_ts, sigmask);
  if (fds && nfds) write_pollfd(ctx, fds, nfds);
  return res;
}
#define INIT_PPOLL COMMON_INTERCEPT_FUNCTION(ppoll);
#else
#define INIT_PPOLL
#endif

#if SANITIZER_INTERCEPT_WORDEXP
INTERCEPTOR(int, wordexp, char *s, __sanitizer_wordexp_t *p, int flags) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, wordexp, s, p, flags);
  if (s) COMMON_INTERCEPTOR_READ_RANGE(ctx, s, REAL(strlen)(s) + 1);
  int res = REAL(wordexp)(s, p, flags);
  if (!res && p) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p, sizeof(*p));
    if (p->we_wordc)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p->we_wordv,
                                     sizeof(*p->we_wordv) * p->we_wordc);
    for (uptr i = 0; i < p->we_wordc; ++i) {
      char *w = p->we_wordv[i];
      if (w) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, w, REAL(strlen)(w) + 1);
    }
  }
  return res;
}
#define INIT_WORDEXP COMMON_INTERCEPT_FUNCTION(wordexp);
#else
#define INIT_WORDEXP
#endif

#if SANITIZER_INTERCEPT_SIGWAIT
INTERCEPTOR(int, sigwait, __sanitizer_sigset_t *set, int *sig) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sigwait, set, sig);
  // FIXME: read sigset_t when all of sigemptyset, etc are intercepted
  int res = REAL(sigwait)(set, sig);
  if (!res && sig) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, sig, sizeof(*sig));
  return res;
}
#define INIT_SIGWAIT COMMON_INTERCEPT_FUNCTION(sigwait);
#else
#define INIT_SIGWAIT
#endif

#if SANITIZER_INTERCEPT_SIGWAITINFO
INTERCEPTOR(int, sigwaitinfo, __sanitizer_sigset_t *set, void *info) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sigwaitinfo, set, info);
  // FIXME: read sigset_t when all of sigemptyset, etc are intercepted
  int res = REAL(sigwaitinfo)(set, info);
  if (res > 0 && info) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, info, siginfo_t_sz);
  return res;
}
#define INIT_SIGWAITINFO COMMON_INTERCEPT_FUNCTION(sigwaitinfo);
#else
#define INIT_SIGWAITINFO
#endif

#if SANITIZER_INTERCEPT_SIGTIMEDWAIT
INTERCEPTOR(int, sigtimedwait, __sanitizer_sigset_t *set, void *info,
            void *timeout) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sigtimedwait, set, info, timeout);
  if (timeout) COMMON_INTERCEPTOR_READ_RANGE(ctx, timeout, struct_timespec_sz);
  // FIXME: read sigset_t when all of sigemptyset, etc are intercepted
  int res = REAL(sigtimedwait)(set, info, timeout);
  if (res > 0 && info) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, info, siginfo_t_sz);
  return res;
}
#define INIT_SIGTIMEDWAIT COMMON_INTERCEPT_FUNCTION(sigtimedwait);
#else
#define INIT_SIGTIMEDWAIT
#endif

#if SANITIZER_INTERCEPT_SIGSETOPS
INTERCEPTOR(int, sigemptyset, __sanitizer_sigset_t *set) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sigemptyset, set);
  int res = REAL(sigemptyset)(set);
  if (!res && set) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, set, sizeof(*set));
  return res;
}

INTERCEPTOR(int, sigfillset, __sanitizer_sigset_t *set) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sigfillset, set);
  int res = REAL(sigfillset)(set);
  if (!res && set) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, set, sizeof(*set));
  return res;
}
#define INIT_SIGSETOPS                    \
  COMMON_INTERCEPT_FUNCTION(sigemptyset); \
  COMMON_INTERCEPT_FUNCTION(sigfillset);
#else
#define INIT_SIGSETOPS
#endif

#if SANITIZER_INTERCEPT_SIGPENDING
INTERCEPTOR(int, sigpending, __sanitizer_sigset_t *set) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sigpending, set);
  int res = REAL(sigpending)(set);
  if (!res && set) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, set, sizeof(*set));
  return res;
}
#define INIT_SIGPENDING COMMON_INTERCEPT_FUNCTION(sigpending);
#else
#define INIT_SIGPENDING
#endif

#if SANITIZER_INTERCEPT_SIGPROCMASK
INTERCEPTOR(int, sigprocmask, int how, __sanitizer_sigset_t *set,
            __sanitizer_sigset_t *oldset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sigprocmask, how, set, oldset);
  // FIXME: read sigset_t when all of sigemptyset, etc are intercepted
  int res = REAL(sigprocmask)(how, set, oldset);
  if (!res && oldset)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, oldset, sizeof(*oldset));
  return res;
}
#define INIT_SIGPROCMASK COMMON_INTERCEPT_FUNCTION(sigprocmask);
#else
#define INIT_SIGPROCMASK
#endif

#if SANITIZER_INTERCEPT_BACKTRACE
INTERCEPTOR(int, backtrace, void **buffer, int size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, backtrace, buffer, size);
  int res = REAL(backtrace)(buffer, size);
  if (res && buffer)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buffer, res * sizeof(*buffer));
  return res;
}

INTERCEPTOR(char **, backtrace_symbols, void **buffer, int size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, backtrace_symbols, buffer, size);
  if (buffer && size)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, buffer, size * sizeof(*buffer));
  char **res = REAL(backtrace_symbols)(buffer, size);
  if (res && size) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, size * sizeof(*res));
    for (int i = 0; i < size; ++i)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res[i], REAL(strlen(res[i])) + 1);
  }
  return res;
}
#define INIT_BACKTRACE                  \
  COMMON_INTERCEPT_FUNCTION(backtrace); \
  COMMON_INTERCEPT_FUNCTION(backtrace_symbols);
#else
#define INIT_BACKTRACE
#endif

#if SANITIZER_INTERCEPT__EXIT
INTERCEPTOR(void, _exit, int status) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, _exit, status);
  int status1 = COMMON_INTERCEPTOR_ON_EXIT(ctx);
  if (status == 0) status = status1;
  REAL(_exit)(status);
}
#define INIT__EXIT COMMON_INTERCEPT_FUNCTION(_exit);
#else
#define INIT__EXIT
#endif

#if SANITIZER_INTERCEPT_PHTREAD_MUTEX
INTERCEPTOR(int, pthread_mutex_lock, void *m) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_mutex_lock, m);
  int res = REAL(pthread_mutex_lock)(m);
  if (res == errno_EOWNERDEAD)
    COMMON_INTERCEPTOR_MUTEX_REPAIR(ctx, m);
  if (res == 0 || res == errno_EOWNERDEAD)
    COMMON_INTERCEPTOR_MUTEX_LOCK(ctx, m);
  return res;
}

INTERCEPTOR(int, pthread_mutex_unlock, void *m) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_mutex_unlock, m);
  COMMON_INTERCEPTOR_MUTEX_UNLOCK(ctx, m);
  return REAL(pthread_mutex_unlock)(m);
}

#define INIT_PTHREAD_MUTEX_LOCK COMMON_INTERCEPT_FUNCTION(pthread_mutex_lock)
#define INIT_PTHREAD_MUTEX_UNLOCK \
  COMMON_INTERCEPT_FUNCTION(pthread_mutex_unlock)
#else
#define INIT_PTHREAD_MUTEX_LOCK
#define INIT_PTHREAD_MUTEX_UNLOCK
#endif

#if SANITIZER_INTERCEPT_PTHREAD_COND
INTERCEPTOR(int, pthread_cond_wait, void *c, void *m) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_cond_wait, c, m);
  COMMON_INTERCEPTOR_MUTEX_UNLOCK(ctx, m);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, c, pthread_cond_t_sz);
  int res = REAL(pthread_cond_wait)(c, m);
  COMMON_INTERCEPTOR_MUTEX_LOCK(ctx, m);
  return res;
}

INTERCEPTOR(int, pthread_cond_init, void *c, void *a) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_cond_init, c, a);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, c, pthread_cond_t_sz);
  return REAL(pthread_cond_init)(c, a);
}

INTERCEPTOR(int, pthread_cond_signal, void *c) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_cond_signal, c);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, c, pthread_cond_t_sz);
  return REAL(pthread_cond_signal)(c);
}

INTERCEPTOR(int, pthread_cond_broadcast, void *c) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_cond_broadcast, c);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, c, pthread_cond_t_sz);
  return REAL(pthread_cond_broadcast)(c);
}

#define INIT_PTHREAD_COND_WAIT \
  INTERCEPT_FUNCTION_VER(pthread_cond_wait, "GLIBC_2.3.2")
#define INIT_PTHREAD_COND_INIT \
  INTERCEPT_FUNCTION_VER(pthread_cond_init, "GLIBC_2.3.2")
#define INIT_PTHREAD_COND_SIGNAL \
  INTERCEPT_FUNCTION_VER(pthread_cond_signal, "GLIBC_2.3.2")
#define INIT_PTHREAD_COND_BROADCAST \
  INTERCEPT_FUNCTION_VER(pthread_cond_broadcast, "GLIBC_2.3.2")
#else
#define INIT_PTHREAD_COND_WAIT
#define INIT_PTHREAD_COND_INIT
#define INIT_PTHREAD_COND_SIGNAL
#define INIT_PTHREAD_COND_BROADCAST
#endif

#if SANITIZER_INTERCEPT_GETMNTENT || SANITIZER_INTERCEPT_GETMNTENT_R
static void write_mntent(void *ctx, __sanitizer_mntent *mnt) {
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, mnt, sizeof(*mnt));
  if (mnt->mnt_fsname)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, mnt->mnt_fsname,
                                   REAL(strlen)(mnt->mnt_fsname) + 1);
  if (mnt->mnt_dir)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, mnt->mnt_dir,
                                   REAL(strlen)(mnt->mnt_dir) + 1);
  if (mnt->mnt_type)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, mnt->mnt_type,
                                   REAL(strlen)(mnt->mnt_type) + 1);
  if (mnt->mnt_opts)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, mnt->mnt_opts,
                                   REAL(strlen)(mnt->mnt_opts) + 1);
}
#endif

#if SANITIZER_INTERCEPT_GETMNTENT
INTERCEPTOR(__sanitizer_mntent *, getmntent, void *fp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getmntent, fp);
  __sanitizer_mntent *res = REAL(getmntent)(fp);
  if (res) write_mntent(ctx, res);
  return res;
}
#define INIT_GETMNTENT COMMON_INTERCEPT_FUNCTION(getmntent);
#else
#define INIT_GETMNTENT
#endif

#if SANITIZER_INTERCEPT_GETMNTENT_R
INTERCEPTOR(__sanitizer_mntent *, getmntent_r, void *fp,
            __sanitizer_mntent *mntbuf, char *buf, int buflen) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getmntent_r, fp, mntbuf, buf, buflen);
  __sanitizer_mntent *res = REAL(getmntent_r)(fp, mntbuf, buf, buflen);
  if (res) write_mntent(ctx, res);
  return res;
}
#define INIT_GETMNTENT_R COMMON_INTERCEPT_FUNCTION(getmntent_r);
#else
#define INIT_GETMNTENT_R
#endif

#if SANITIZER_INTERCEPT_STATFS
INTERCEPTOR(int, statfs, char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, statfs, path, buf);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  int res = REAL(statfs)(path, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statfs_sz);
  return res;
}
INTERCEPTOR(int, fstatfs, int fd, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fstatfs, fd, buf);
  int res = REAL(fstatfs)(fd, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statfs_sz);
  return res;
}
#define INIT_STATFS                  \
  COMMON_INTERCEPT_FUNCTION(statfs); \
  COMMON_INTERCEPT_FUNCTION(fstatfs);
#else
#define INIT_STATFS
#endif

#if SANITIZER_INTERCEPT_STATFS64
INTERCEPTOR(int, statfs64, char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, statfs64, path, buf);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  int res = REAL(statfs64)(path, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statfs64_sz);
  return res;
}
INTERCEPTOR(int, fstatfs64, int fd, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fstatfs64, fd, buf);
  int res = REAL(fstatfs64)(fd, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statfs64_sz);
  return res;
}
#define INIT_STATFS64                  \
  COMMON_INTERCEPT_FUNCTION(statfs64); \
  COMMON_INTERCEPT_FUNCTION(fstatfs64);
#else
#define INIT_STATFS64
#endif

#if SANITIZER_INTERCEPT_STATVFS
INTERCEPTOR(int, statvfs, char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, statvfs, path, buf);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  int res = REAL(statvfs)(path, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statvfs_sz);
  return res;
}
INTERCEPTOR(int, fstatvfs, int fd, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fstatvfs, fd, buf);
  int res = REAL(fstatvfs)(fd, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statvfs_sz);
  return res;
}
#define INIT_STATVFS                  \
  COMMON_INTERCEPT_FUNCTION(statvfs); \
  COMMON_INTERCEPT_FUNCTION(fstatvfs);
#else
#define INIT_STATVFS
#endif

#if SANITIZER_INTERCEPT_STATVFS64
INTERCEPTOR(int, statvfs64, char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, statvfs64, path, buf);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  int res = REAL(statvfs64)(path, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statvfs64_sz);
  return res;
}
INTERCEPTOR(int, fstatvfs64, int fd, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fstatvfs64, fd, buf);
  int res = REAL(fstatvfs64)(fd, buf);
  if (!res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, struct_statvfs64_sz);
  return res;
}
#define INIT_STATVFS64                  \
  COMMON_INTERCEPT_FUNCTION(statvfs64); \
  COMMON_INTERCEPT_FUNCTION(fstatvfs64);
#else
#define INIT_STATVFS64
#endif

#if SANITIZER_INTERCEPT_INITGROUPS
INTERCEPTOR(int, initgroups, char *user, u32 group) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, initgroups, user, group);
  if (user) COMMON_INTERCEPTOR_READ_RANGE(ctx, user, REAL(strlen)(user) + 1);
  int res = REAL(initgroups)(user, group);
  return res;
}
#define INIT_INITGROUPS COMMON_INTERCEPT_FUNCTION(initgroups);
#else
#define INIT_INITGROUPS
#endif

#if SANITIZER_INTERCEPT_ETHER
INTERCEPTOR(char *, ether_ntoa, __sanitizer_ether_addr *addr) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ether_ntoa, addr);
  if (addr) COMMON_INTERCEPTOR_READ_RANGE(ctx, addr, sizeof(*addr));
  char *res = REAL(ether_ntoa)(addr);
  if (res) COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
INTERCEPTOR(__sanitizer_ether_addr *, ether_aton, char *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ether_aton, buf);
  if (buf) COMMON_INTERCEPTOR_READ_RANGE(ctx, buf, REAL(strlen)(buf) + 1);
  __sanitizer_ether_addr *res = REAL(ether_aton)(buf);
  if (res) COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, res, sizeof(*res));
  return res;
}
INTERCEPTOR(int, ether_ntohost, char *hostname, __sanitizer_ether_addr *addr) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ether_ntohost, hostname, addr);
  if (addr) COMMON_INTERCEPTOR_READ_RANGE(ctx, addr, sizeof(*addr));
  int res = REAL(ether_ntohost)(hostname, addr);
  if (!res && hostname)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, hostname, REAL(strlen)(hostname) + 1);
  return res;
}
INTERCEPTOR(int, ether_hostton, char *hostname, __sanitizer_ether_addr *addr) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ether_hostton, hostname, addr);
  if (hostname)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, hostname, REAL(strlen)(hostname) + 1);
  int res = REAL(ether_hostton)(hostname, addr);
  if (!res && addr) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, sizeof(*addr));
  return res;
}
INTERCEPTOR(int, ether_line, char *line, __sanitizer_ether_addr *addr,
            char *hostname) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ether_line, line, addr, hostname);
  if (line) COMMON_INTERCEPTOR_READ_RANGE(ctx, line, REAL(strlen)(line) + 1);
  int res = REAL(ether_line)(line, addr, hostname);
  if (!res) {
    if (addr) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, sizeof(*addr));
    if (hostname)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, hostname, REAL(strlen)(hostname) + 1);
  }
  return res;
}
#define INIT_ETHER                          \
  COMMON_INTERCEPT_FUNCTION(ether_ntoa);    \
  COMMON_INTERCEPT_FUNCTION(ether_aton);    \
  COMMON_INTERCEPT_FUNCTION(ether_ntohost); \
  COMMON_INTERCEPT_FUNCTION(ether_hostton); \
  COMMON_INTERCEPT_FUNCTION(ether_line);
#else
#define INIT_ETHER
#endif

#if SANITIZER_INTERCEPT_ETHER_R
INTERCEPTOR(char *, ether_ntoa_r, __sanitizer_ether_addr *addr, char *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ether_ntoa_r, addr, buf);
  if (addr) COMMON_INTERCEPTOR_READ_RANGE(ctx, addr, sizeof(*addr));
  char *res = REAL(ether_ntoa_r)(addr, buf);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
INTERCEPTOR(__sanitizer_ether_addr *, ether_aton_r, char *buf,
            __sanitizer_ether_addr *addr) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, ether_aton_r, buf, addr);
  if (buf) COMMON_INTERCEPTOR_READ_RANGE(ctx, buf, REAL(strlen)(buf) + 1);
  __sanitizer_ether_addr *res = REAL(ether_aton_r)(buf, addr);
  if (res) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, res, sizeof(*res));
  return res;
}
#define INIT_ETHER_R                       \
  COMMON_INTERCEPT_FUNCTION(ether_ntoa_r); \
  COMMON_INTERCEPT_FUNCTION(ether_aton_r);
#else
#define INIT_ETHER_R
#endif

#if SANITIZER_INTERCEPT_SHMCTL
INTERCEPTOR(int, shmctl, int shmid, int cmd, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, shmctl, shmid, cmd, buf);
  int res = REAL(shmctl)(shmid, cmd, buf);
  if (res >= 0) {
    unsigned sz = 0;
    if (cmd == shmctl_ipc_stat || cmd == shmctl_shm_stat)
      sz = sizeof(__sanitizer_shmid_ds);
    else if (cmd == shmctl_ipc_info)
      sz = struct_shminfo_sz;
    else if (cmd == shmctl_shm_info)
      sz = struct_shm_info_sz;
    if (sz) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, buf, sz);
  }
  return res;
}
#define INIT_SHMCTL COMMON_INTERCEPT_FUNCTION(shmctl);
#else
#define INIT_SHMCTL
#endif

#if SANITIZER_INTERCEPT_RANDOM_R
INTERCEPTOR(int, random_r, void *buf, u32 *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, random_r, buf, result);
  int res = REAL(random_r)(buf, result);
  if (!res && result)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
  return res;
}
#define INIT_RANDOM_R COMMON_INTERCEPT_FUNCTION(random_r);
#else
#define INIT_RANDOM_R
#endif

#if SANITIZER_INTERCEPT_PTHREAD_ATTR_GET || \
    SANITIZER_INTERCEPT_PTHREAD_ATTR_GETINHERITSSCHED
#define INTERCEPTOR_PTHREAD_ATTR_GET(what, sz)                      \
  INTERCEPTOR(int, pthread_attr_get##what, void *attr, void *r) {   \
    void *ctx;                                                      \
    COMMON_INTERCEPTOR_ENTER(ctx, pthread_attr_get##what, attr, r); \
    int res = REAL(pthread_attr_get##what)(attr, r);                \
    if (!res && r) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, r, sz);      \
    return res;                                                     \
  }
#endif

#if SANITIZER_INTERCEPT_PTHREAD_ATTR_GET
INTERCEPTOR_PTHREAD_ATTR_GET(detachstate, sizeof(int))
INTERCEPTOR_PTHREAD_ATTR_GET(guardsize, sizeof(SIZE_T))
INTERCEPTOR_PTHREAD_ATTR_GET(schedparam, struct_sched_param_sz)
INTERCEPTOR_PTHREAD_ATTR_GET(schedpolicy, sizeof(int))
INTERCEPTOR_PTHREAD_ATTR_GET(scope, sizeof(int))
INTERCEPTOR_PTHREAD_ATTR_GET(stacksize, sizeof(SIZE_T))
INTERCEPTOR(int, pthread_attr_getstack, void *attr, void **addr, SIZE_T *size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_attr_getstack, attr, addr, size);
  int res = REAL(pthread_attr_getstack)(attr, addr, size);
  if (!res) {
    if (addr) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, addr, sizeof(*addr));
    if (size) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, size, sizeof(*size));
  }
  return res;
}

// We may need to call the real pthread_attr_getstack from the run-time
// in sanitizer_common, but we don't want to include the interception headers
// there. So, just define this function here.
namespace __sanitizer {
int real_pthread_attr_getstack(void *attr, void **addr, SIZE_T *size) {
  return REAL(pthread_attr_getstack)(attr, addr, size);
}
}  // namespace __sanitizer

#define INIT_PTHREAD_ATTR_GET                             \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getdetachstate); \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getguardsize);   \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getschedparam);  \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getschedpolicy); \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getscope);       \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getstacksize);   \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getstack);
#else
#define INIT_PTHREAD_ATTR_GET
#endif

#if SANITIZER_INTERCEPT_PTHREAD_ATTR_GETINHERITSCHED
INTERCEPTOR_PTHREAD_ATTR_GET(inheritsched, sizeof(int))

#define INIT_PTHREAD_ATTR_GETINHERITSCHED \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getinheritsched);
#else
#define INIT_PTHREAD_ATTR_GETINHERITSCHED
#endif

#if SANITIZER_INTERCEPT_PTHREAD_ATTR_GETAFFINITY_NP
INTERCEPTOR(int, pthread_attr_getaffinity_np, void *attr, SIZE_T cpusetsize,
            void *cpuset) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_attr_getaffinity_np, attr, cpusetsize,
                           cpuset);
  int res = REAL(pthread_attr_getaffinity_np)(attr, cpusetsize, cpuset);
  if (!res && cpusetsize && cpuset)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, cpuset, cpusetsize);
  return res;
}

#define INIT_PTHREAD_ATTR_GETAFFINITY_NP \
  COMMON_INTERCEPT_FUNCTION(pthread_attr_getaffinity_np);
#else
#define INIT_PTHREAD_ATTR_GETAFFINITY_NP
#endif

#if SANITIZER_INTERCEPT_TMPNAM
INTERCEPTOR(char *, tmpnam, char *s) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, tmpnam, s);
  char *res = REAL(tmpnam)(s);
  if (res) {
    if (s)
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, s, REAL(strlen)(s) + 1);
    else
      COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  }
  return res;
}
#define INIT_TMPNAM COMMON_INTERCEPT_FUNCTION(tmpnam);
#else
#define INIT_TMPNAM
#endif

#if SANITIZER_INTERCEPT_TMPNAM_R
INTERCEPTOR(char *, tmpnam_r, char *s) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, tmpnam_r, s);
  char *res = REAL(tmpnam_r)(s);
  if (res && s) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, s, REAL(strlen)(s) + 1);
  return res;
}
#define INIT_TMPNAM_R COMMON_INTERCEPT_FUNCTION(tmpnam_r);
#else
#define INIT_TMPNAM_R
#endif

#if SANITIZER_INTERCEPT_TEMPNAM
INTERCEPTOR(char *, tempnam, char *dir, char *pfx) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, tempnam, dir, pfx);
  if (dir) COMMON_INTERCEPTOR_READ_RANGE(ctx, dir, REAL(strlen)(dir) + 1);
  if (pfx) COMMON_INTERCEPTOR_READ_RANGE(ctx, pfx, REAL(strlen)(pfx) + 1);
  char *res = REAL(tempnam)(dir, pfx);
  if (res) COMMON_INTERCEPTOR_INITIALIZE_RANGE(ctx, res, REAL(strlen)(res) + 1);
  return res;
}
#define INIT_TEMPNAM COMMON_INTERCEPT_FUNCTION(tempnam);
#else
#define INIT_TEMPNAM
#endif

#if SANITIZER_INTERCEPT_PTHREAD_SETNAME_NP
INTERCEPTOR(int, pthread_setname_np, uptr thread, const char *name) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, pthread_setname_np, thread, name);
  COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name);
  return REAL(pthread_setname_np)(thread, name);
}
#define INIT_PTHREAD_SETNAME_NP COMMON_INTERCEPT_FUNCTION(pthread_setname_np);
#else
#define INIT_PTHREAD_SETNAME_NP
#endif

#if SANITIZER_INTERCEPT_SINCOS
INTERCEPTOR(void, sincos, double x, double *sin, double *cos) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sincos, x, sin, cos);
  REAL(sincos)(x, sin, cos);
  if (sin) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, sin, sizeof(*sin));
  if (cos) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, cos, sizeof(*cos));
}
INTERCEPTOR(void, sincosf, float x, float *sin, float *cos) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sincosf, x, sin, cos);
  REAL(sincosf)(x, sin, cos);
  if (sin) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, sin, sizeof(*sin));
  if (cos) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, cos, sizeof(*cos));
}
INTERCEPTOR(void, sincosl, long double x, long double *sin, long double *cos) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, sincosl, x, sin, cos);
  REAL(sincosl)(x, sin, cos);
  if (sin) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, sin, sizeof(*sin));
  if (cos) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, cos, sizeof(*cos));
}
#define INIT_SINCOS                   \
  COMMON_INTERCEPT_FUNCTION(sincos);  \
  COMMON_INTERCEPT_FUNCTION(sincosf); \
  COMMON_INTERCEPT_FUNCTION(sincosl);
#else
#define INIT_SINCOS
#endif

#if SANITIZER_INTERCEPT_REMQUO
INTERCEPTOR(double, remquo, double x, double y, int *quo) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, remquo, x, y, quo);
  double res = REAL(remquo)(x, y, quo);
  if (quo) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, quo, sizeof(*quo));
  return res;
}
INTERCEPTOR(float, remquof, float x, float y, int *quo) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, remquof, x, y, quo);
  float res = REAL(remquof)(x, y, quo);
  if (quo) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, quo, sizeof(*quo));
  return res;
}
INTERCEPTOR(long double, remquol, long double x, long double y, int *quo) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, remquol, x, y, quo);
  long double res = REAL(remquol)(x, y, quo);
  if (quo) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, quo, sizeof(*quo));
  return res;
}
#define INIT_REMQUO                   \
  COMMON_INTERCEPT_FUNCTION(remquo);  \
  COMMON_INTERCEPT_FUNCTION(remquof); \
  COMMON_INTERCEPT_FUNCTION(remquol);
#else
#define INIT_REMQUO
#endif

#if SANITIZER_INTERCEPT_LGAMMA
extern int signgam;
INTERCEPTOR(double, lgamma, double x) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lgamma, x);
  double res = REAL(lgamma)(x);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, &signgam, sizeof(signgam));
  return res;
}
INTERCEPTOR(float, lgammaf, float x) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lgammaf, x);
  float res = REAL(lgammaf)(x);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, &signgam, sizeof(signgam));
  return res;
}
INTERCEPTOR(long double, lgammal, long double x) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lgammal, x);
  long double res = REAL(lgammal)(x);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, &signgam, sizeof(signgam));
  return res;
}
#define INIT_LGAMMA                   \
  COMMON_INTERCEPT_FUNCTION(lgamma);  \
  COMMON_INTERCEPT_FUNCTION(lgammaf); \
  COMMON_INTERCEPT_FUNCTION(lgammal);
#else
#define INIT_LGAMMA
#endif

#if SANITIZER_INTERCEPT_LGAMMA_R
INTERCEPTOR(double, lgamma_r, double x, int *signp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lgamma_r, x, signp);
  double res = REAL(lgamma_r)(x, signp);
  if (signp) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, signp, sizeof(*signp));
  return res;
}
INTERCEPTOR(float, lgammaf_r, float x, int *signp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lgammaf_r, x, signp);
  float res = REAL(lgammaf_r)(x, signp);
  if (signp) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, signp, sizeof(*signp));
  return res;
}
INTERCEPTOR(long double, lgammal_r, long double x, int *signp) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lgammal_r, x, signp);
  long double res = REAL(lgammal_r)(x, signp);
  if (signp) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, signp, sizeof(*signp));
  return res;
}
#define INIT_LGAMMA_R                   \
  COMMON_INTERCEPT_FUNCTION(lgamma_r);  \
  COMMON_INTERCEPT_FUNCTION(lgammaf_r); \
  COMMON_INTERCEPT_FUNCTION(lgammal_r);
#else
#define INIT_LGAMMA_R
#endif

#if SANITIZER_INTERCEPT_DRAND48_R
INTERCEPTOR(int, drand48_r, void *buffer, double *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, drand48_r, buffer, result);
  int res = REAL(drand48_r)(buffer, result);
  if (result) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
  return res;
}
INTERCEPTOR(int, lrand48_r, void *buffer, long *result) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lrand48_r, buffer, result);
  int res = REAL(lrand48_r)(buffer, result);
  if (result) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, result, sizeof(*result));
  return res;
}
#define INIT_DRAND48_R                  \
  COMMON_INTERCEPT_FUNCTION(drand48_r); \
  COMMON_INTERCEPT_FUNCTION(lrand48_r);
#else
#define INIT_DRAND48_R
#endif

#if SANITIZER_INTERCEPT_GETLINE
INTERCEPTOR(SSIZE_T, getline, char **lineptr, SIZE_T *n, void *stream) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getline, lineptr, n, stream);
  SSIZE_T res = REAL(getline)(lineptr, n, stream);
  if (res > 0) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, lineptr, sizeof(*lineptr));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, n, sizeof(*n));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *lineptr, res + 1);
  }
  return res;
}
INTERCEPTOR(SSIZE_T, getdelim, char **lineptr, SIZE_T *n, int delim,
            void *stream) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getdelim, lineptr, n, delim, stream);
  SSIZE_T res = REAL(getdelim)(lineptr, n, delim, stream);
  if (res > 0) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, lineptr, sizeof(*lineptr));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, n, sizeof(*n));
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, *lineptr, res + 1);
  }
  return res;
}
#define INIT_GETLINE                  \
  COMMON_INTERCEPT_FUNCTION(getline); \
  COMMON_INTERCEPT_FUNCTION(getdelim);
#else
#define INIT_GETLINE
#endif

#if SANITIZER_INTERCEPT_ICONV
INTERCEPTOR(SIZE_T, iconv, void *cd, char **inbuf, SIZE_T *inbytesleft,
            char **outbuf, SIZE_T *outbytesleft) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, iconv, cd, inbuf, inbytesleft, outbuf,
                           outbytesleft);
  if (inbytesleft)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, inbytesleft, sizeof(*inbytesleft));
  if (inbuf && inbytesleft)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, *inbuf, *inbytesleft);
  if (outbytesleft)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, outbytesleft, sizeof(*outbytesleft));
  void *outbuf_orig = outbuf ? *outbuf : 0;
  SIZE_T res = REAL(iconv)(cd, inbuf, inbytesleft, outbuf, outbytesleft);
  if (res != (SIZE_T) - 1 && outbuf && *outbuf > outbuf_orig) {
    SIZE_T sz = (char *)*outbuf - (char *)outbuf_orig;
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, outbuf_orig, sz);
  }
  return res;
}
#define INIT_ICONV COMMON_INTERCEPT_FUNCTION(iconv);
#else
#define INIT_ICONV
#endif

#if SANITIZER_INTERCEPT_TIMES
INTERCEPTOR(__sanitizer_clock_t, times, void *tms) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, times, tms);
  __sanitizer_clock_t res = REAL(times)(tms);
  if (res != (__sanitizer_clock_t)-1 && tms)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, tms, struct_tms_sz);
  return res;
}
#define INIT_TIMES COMMON_INTERCEPT_FUNCTION(times);
#else
#define INIT_TIMES
#endif

#if SANITIZER_INTERCEPT_TLS_GET_ADDR
#define INIT_TLS_GET_ADDR COMMON_INTERCEPT_FUNCTION(__tls_get_addr)
INTERCEPTOR(void *, __tls_get_addr, void *arg) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, __tls_get_addr, arg);
  void *res = REAL(__tls_get_addr)(arg);
  DTLS_on_tls_get_addr(arg, res);
  return res;
}
#else
#define INIT_TLS_GET_ADDR
#endif

#if SANITIZER_INTERCEPT_LISTXATTR
INTERCEPTOR(SSIZE_T, listxattr, const char *path, char *list, SIZE_T size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, listxattr, path, list, size);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  SSIZE_T res = REAL(listxattr)(path, list, size);
  // Here and below, size == 0 is a special case where nothing is written to the
  // buffer, and res contains the desired buffer size.
  if (size && res > 0 && list) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, list, res);
  return res;
}
INTERCEPTOR(SSIZE_T, llistxattr, const char *path, char *list, SIZE_T size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, llistxattr, path, list, size);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  SSIZE_T res = REAL(llistxattr)(path, list, size);
  if (size && res > 0 && list) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, list, res);
  return res;
}
INTERCEPTOR(SSIZE_T, flistxattr, int fd, char *list, SIZE_T size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, flistxattr, fd, list, size);
  SSIZE_T res = REAL(flistxattr)(fd, list, size);
  if (size && res > 0 && list) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, list, res);
  return res;
}
#define INIT_LISTXATTR                   \
  COMMON_INTERCEPT_FUNCTION(listxattr);  \
  COMMON_INTERCEPT_FUNCTION(llistxattr); \
  COMMON_INTERCEPT_FUNCTION(flistxattr);
#else
#define INIT_LISTXATTR
#endif

#if SANITIZER_INTERCEPT_GETXATTR
INTERCEPTOR(SSIZE_T, getxattr, const char *path, const char *name, char *value,
            SIZE_T size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getxattr, path, name, value, size);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  if (name) COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  SSIZE_T res = REAL(getxattr)(path, name, value, size);
  if (size && res > 0 && value) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, value, res);
  return res;
}
INTERCEPTOR(SSIZE_T, lgetxattr, const char *path, const char *name, char *value,
            SIZE_T size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lgetxattr, path, name, value, size);
  if (path) COMMON_INTERCEPTOR_READ_RANGE(ctx, path, REAL(strlen)(path) + 1);
  if (name) COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  SSIZE_T res = REAL(lgetxattr)(path, name, value, size);
  if (size && res > 0 && value) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, value, res);
  return res;
}
INTERCEPTOR(SSIZE_T, fgetxattr, int fd, const char *name, char *value,
            SIZE_T size) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fgetxattr, fd, name, value, size);
  if (name) COMMON_INTERCEPTOR_READ_RANGE(ctx, name, REAL(strlen)(name) + 1);
  SSIZE_T res = REAL(fgetxattr)(fd, name, value, size);
  if (size && res > 0 && value) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, value, res);
  return res;
}
#define INIT_GETXATTR                   \
  COMMON_INTERCEPT_FUNCTION(getxattr);  \
  COMMON_INTERCEPT_FUNCTION(lgetxattr); \
  COMMON_INTERCEPT_FUNCTION(fgetxattr);
#else
#define INIT_GETXATTR
#endif

#if SANITIZER_INTERCEPT_GETRESID
INTERCEPTOR(int, getresuid, void *ruid, void *euid, void *suid) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getresuid, ruid, euid, suid);
  int res = REAL(getresuid)(ruid, euid, suid);
  if (res >= 0) {
    if (ruid) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ruid, uid_t_sz);
    if (euid) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, euid, uid_t_sz);
    if (suid) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, suid, uid_t_sz);
  }
  return res;
}
INTERCEPTOR(int, getresgid, void *rgid, void *egid, void *sgid) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getresgid, rgid, egid, sgid);
  int res = REAL(getresgid)(rgid, egid, sgid);
  if (res >= 0) {
    if (rgid) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, rgid, gid_t_sz);
    if (egid) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, egid, gid_t_sz);
    if (sgid) COMMON_INTERCEPTOR_WRITE_RANGE(ctx, sgid, gid_t_sz);
  }
  return res;
}
#define INIT_GETRESID                   \
  COMMON_INTERCEPT_FUNCTION(getresuid); \
  COMMON_INTERCEPT_FUNCTION(getresgid);
#else
#define INIT_GETRESID
#endif

#if SANITIZER_INTERCEPT_GETIFADDRS
// As long as getifaddrs()/freeifaddrs() use calloc()/free(), we don't need to
// intercept freeifaddrs(). If that ceases to be the case, we might need to
// intercept it to poison the memory again.
INTERCEPTOR(int, getifaddrs, __sanitizer_ifaddrs **ifap) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, getifaddrs, ifap);
  int res = REAL(getifaddrs)(ifap);
  if (res == 0 && ifap) {
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ifap, sizeof(void *));
    __sanitizer_ifaddrs *p = *ifap;
    while (p) {
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p, sizeof(__sanitizer_ifaddrs));
      if (p->ifa_name)
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p->ifa_name,
                                       REAL(strlen)(p->ifa_name) + 1);
      if (p->ifa_addr)
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p->ifa_addr, struct_sockaddr_sz);
      if (p->ifa_netmask)
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p->ifa_netmask, struct_sockaddr_sz);
      // On Linux this is a union, but the other member also points to a
      // struct sockaddr, so the following is sufficient.
      if (p->ifa_dstaddr)
        COMMON_INTERCEPTOR_WRITE_RANGE(ctx, p->ifa_dstaddr, struct_sockaddr_sz);
      // FIXME(smatveev): Unpoison p->ifa_data as well.
      p = p->ifa_next;
    }
  }
  return res;
}
#define INIT_GETIFADDRS                  \
  COMMON_INTERCEPT_FUNCTION(getifaddrs);
#else
#define INIT_GETIFADDRS
#endif

#if SANITIZER_INTERCEPT_IF_INDEXTONAME
INTERCEPTOR(char *, if_indextoname, unsigned int ifindex, char* ifname) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, if_indextoname, ifindex, ifname);
  char *res = REAL(if_indextoname)(ifindex, ifname);
  if (res && ifname)
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ifname, REAL(strlen)(ifname) + 1);
  return res;
}
INTERCEPTOR(unsigned int, if_nametoindex, const char* ifname) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, if_nametoindex, ifname);
  if (ifname)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, ifname, REAL(strlen)(ifname) + 1);
  return REAL(if_nametoindex)(ifname);
}
#define INIT_IF_INDEXTONAME                  \
  COMMON_INTERCEPT_FUNCTION(if_indextoname); \
  COMMON_INTERCEPT_FUNCTION(if_nametoindex);
#else
#define INIT_IF_INDEXTONAME
#endif

#define SANITIZER_COMMON_INTERCEPTORS_INIT \
  INIT_TEXTDOMAIN;                         \
  INIT_STRCMP;                             \
  INIT_STRNCMP;                            \
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
  INIT_STRPTIME;                           \
  INIT_SCANF;                              \
  INIT_ISOC99_SCANF;                       \
  INIT_PRINTF;                             \
  INIT_ISOC99_PRINTF;                      \
  INIT_FREXP;                              \
  INIT_FREXPF_FREXPL;                      \
  INIT_GETPWNAM_AND_FRIENDS;               \
  INIT_GETPWNAM_R_AND_FRIENDS;             \
  INIT_CLOCK_GETTIME;                      \
  INIT_GETITIMER;                          \
  INIT_TIME;                               \
  INIT_GLOB;                               \
  INIT_WAIT;                               \
  INIT_WAIT4;                              \
  INIT_INET;                               \
  INIT_PTHREAD_GETSCHEDPARAM;              \
  INIT_GETADDRINFO;                        \
  INIT_GETNAMEINFO;                        \
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
  INIT_READDIR64;                          \
  INIT_PTRACE;                             \
  INIT_SETLOCALE;                          \
  INIT_GETCWD;                             \
  INIT_GET_CURRENT_DIR_NAME;               \
  INIT_STRTOIMAX;                          \
  INIT_MBSTOWCS;                           \
  INIT_MBSNRTOWCS;                         \
  INIT_WCSTOMBS;                           \
  INIT_WCSNRTOMBS;                         \
  INIT_TCGETATTR;                          \
  INIT_REALPATH;                           \
  INIT_CANONICALIZE_FILE_NAME;             \
  INIT_CONFSTR;                            \
  INIT_SCHED_GETAFFINITY;                  \
  INIT_STRERROR;                           \
  INIT_STRERROR_R;                         \
  INIT_XPG_STRERROR_R;                     \
  INIT_SCANDIR;                            \
  INIT_SCANDIR64;                          \
  INIT_GETGROUPS;                          \
  INIT_POLL;                               \
  INIT_PPOLL;                              \
  INIT_WORDEXP;                            \
  INIT_SIGWAIT;                            \
  INIT_SIGWAITINFO;                        \
  INIT_SIGTIMEDWAIT;                       \
  INIT_SIGSETOPS;                          \
  INIT_SIGPENDING;                         \
  INIT_SIGPROCMASK;                        \
  INIT_BACKTRACE;                          \
  INIT__EXIT;                              \
  INIT_PTHREAD_MUTEX_LOCK;                 \
  INIT_PTHREAD_MUTEX_UNLOCK;               \
  INIT_PTHREAD_COND_WAIT;                  \
  INIT_PTHREAD_COND_INIT;                  \
  INIT_PTHREAD_COND_SIGNAL;                \
  INIT_PTHREAD_COND_BROADCAST;             \
  INIT_GETMNTENT;                          \
  INIT_GETMNTENT_R;                        \
  INIT_STATFS;                             \
  INIT_STATFS64;                           \
  INIT_STATVFS;                            \
  INIT_STATVFS64;                          \
  INIT_INITGROUPS;                         \
  INIT_ETHER;                              \
  INIT_ETHER_R;                            \
  INIT_SHMCTL;                             \
  INIT_RANDOM_R;                           \
  INIT_PTHREAD_ATTR_GET;                   \
  INIT_PTHREAD_ATTR_GETINHERITSCHED;       \
  INIT_PTHREAD_ATTR_GETAFFINITY_NP;        \
  INIT_TMPNAM;                             \
  INIT_TMPNAM_R;                           \
  INIT_TEMPNAM;                            \
  INIT_PTHREAD_SETNAME_NP;                 \
  INIT_SINCOS;                             \
  INIT_REMQUO;                             \
  INIT_LGAMMA;                             \
  INIT_LGAMMA_R;                           \
  INIT_DRAND48_R;                          \
  INIT_GETLINE;                            \
  INIT_ICONV;                              \
  INIT_TIMES;                              \
  INIT_TLS_GET_ADDR;                       \
  INIT_LISTXATTR;                          \
  INIT_GETXATTR;                           \
  INIT_GETRESID;                           \
  INIT_GETIFADDRS;                         \
  INIT_IF_INDEXTONAME;
/**/
