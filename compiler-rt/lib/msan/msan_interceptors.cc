//===-- msan_interceptors.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Interceptors for standard library functions.
//
// FIXME: move as many interceptors as possible into
// sanitizer_common/sanitizer_common_interceptors.h
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "msan.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_linux.h"

#include <stdarg.h>
// ACHTUNG! No other system header includes in this file.
// Ideally, we should get rid of stdarg.h as well.

using namespace __msan;

// True if this is a nested interceptor.
static THREADLOCAL int in_interceptor_scope;

struct InterceptorScope {
  InterceptorScope() { ++in_interceptor_scope; }
  ~InterceptorScope() { --in_interceptor_scope; }
};

bool IsInInterceptorScope() {
  return in_interceptor_scope;
}

#define ENSURE_MSAN_INITED() do { \
  CHECK(!msan_init_is_running); \
  if (!msan_inited) { \
    __msan_init(); \
  } \
} while (0)

// Check that [x, x+n) range is unpoisoned.
#define CHECK_UNPOISONED_0(x, n)                                             \
  do {                                                                       \
    sptr offset = __msan_test_shadow(x, n);                                  \
    if (__msan::IsInSymbolizer()) break;                                     \
    if (offset >= 0 && __msan::flags()->report_umrs) {                       \
      GET_CALLER_PC_BP_SP;                                                   \
      (void) sp;                                                             \
      Printf("UMR in %s at offset %d inside [%p, +%d) \n", __FUNCTION__,     \
             offset, x, n);                                                  \
      __msan::PrintWarningWithOrigin(pc, bp,                                 \
                                     __msan_get_origin((char *)x + offset)); \
      if (!__msan::flags()->keep_going) {                                    \
        Printf("Exiting\n");                                                 \
        Die();                                                               \
      }                                                                      \
    }                                                                        \
  } while (0)

// Check that [x, x+n) range is unpoisoned unless we are in a nested
// interceptor.
#define CHECK_UNPOISONED(x, n)                             \
  do {                                                     \
    if (!IsInInterceptorScope()) CHECK_UNPOISONED_0(x, n); \
  } while (0);

static void *fast_memset(void *ptr, int c, SIZE_T n);
static void *fast_memcpy(void *dst, const void *src, SIZE_T n);

INTERCEPTOR(SIZE_T, fread, void *ptr, SIZE_T size, SIZE_T nmemb, void *file) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(fread)(ptr, size, nmemb, file);
  if (res > 0)
    __msan_unpoison(ptr, res *size);
  return res;
}

INTERCEPTOR(SIZE_T, fread_unlocked, void *ptr, SIZE_T size, SIZE_T nmemb,
            void *file) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(fread_unlocked)(ptr, size, nmemb, file);
  if (res > 0)
    __msan_unpoison(ptr, res *size);
  return res;
}

INTERCEPTOR(SSIZE_T, readlink, const char *path, char *buf, SIZE_T bufsiz) {
  ENSURE_MSAN_INITED();
  SSIZE_T res = REAL(readlink)(path, buf, bufsiz);
  if (res > 0)
    __msan_unpoison(buf, res);
  return res;
}

INTERCEPTOR(void *, memcpy, void *dest, const void *src, SIZE_T n) {
  return __msan_memcpy(dest, src, n);
}

INTERCEPTOR(void *, mempcpy, void *dest, const void *src, SIZE_T n) {
  return (char *)__msan_memcpy(dest, src, n) + n;
}

INTERCEPTOR(void *, memmove, void *dest, const void *src, SIZE_T n) {
  return __msan_memmove(dest, src, n);
}

INTERCEPTOR(void *, memset, void *s, int c, SIZE_T n) {
  return __msan_memset(s, c, n);
}

INTERCEPTOR(void *, bcopy, const void *src, void *dest, SIZE_T n) {
  return __msan_memmove(dest, src, n);
}

INTERCEPTOR(int, posix_memalign, void **memptr, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  CHECK_EQ(alignment & (alignment - 1), 0);
  CHECK_NE(memptr, 0);
  *memptr = MsanReallocate(&stack, 0, size, alignment, false);
  CHECK_NE(*memptr, 0);
  __msan_unpoison(memptr, sizeof(*memptr));
  return 0;
}

INTERCEPTOR(void, free, void *ptr) {
  ENSURE_MSAN_INITED();
  if (ptr == 0) return;
  MsanDeallocate(ptr);
}

INTERCEPTOR(SIZE_T, strlen, const char *s) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(strlen)(s);
  CHECK_UNPOISONED(s, res + 1);
  return res;
}

INTERCEPTOR(SIZE_T, strnlen, const char *s, SIZE_T n) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(strnlen)(s, n);
  SIZE_T scan_size = (res == n) ? res : res + 1;
  CHECK_UNPOISONED(s, scan_size);
  return res;
}

// FIXME: Add stricter shadow checks in str* interceptors (ex.: strcpy should
// check the shadow of the terminating \0 byte).

INTERCEPTOR(char *, strcpy, char *dest, const char *src) {  // NOLINT
  ENSURE_MSAN_INITED();
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(strcpy)(dest, src);  // NOLINT
  __msan_copy_poison(dest, src, n + 1);
  return res;
}

INTERCEPTOR(char *, strncpy, char *dest, const char *src, SIZE_T n) {  // NOLINT
  ENSURE_MSAN_INITED();
  SIZE_T copy_size = REAL(strnlen)(src, n);
  if (copy_size < n)
    copy_size++;  // trailing \0
  char *res = REAL(strncpy)(dest, src, n);  // NOLINT
  __msan_copy_poison(dest, src, copy_size);
  return res;
}

INTERCEPTOR(char *, stpcpy, char *dest, const char *src) {  // NOLINT
  ENSURE_MSAN_INITED();
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(stpcpy)(dest, src);  // NOLINT
  __msan_copy_poison(dest, src, n + 1);
  return res;
}

INTERCEPTOR(char *, strdup, char *src) {
  ENSURE_MSAN_INITED();
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(strdup)(src);
  __msan_copy_poison(res, src, n + 1);
  return res;
}

INTERCEPTOR(char *, __strdup, char *src) {
  ENSURE_MSAN_INITED();
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(__strdup)(src);
  __msan_copy_poison(res, src, n + 1);
  return res;
}

INTERCEPTOR(char *, strndup, char *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  SIZE_T copy_size = REAL(strnlen)(src, n);
  char *res = REAL(strndup)(src, n);
  __msan_copy_poison(res, src, copy_size);
  __msan_unpoison(res + copy_size, 1); // \0
  return res;
}

INTERCEPTOR(char *, __strndup, char *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  SIZE_T copy_size = REAL(strnlen)(src, n);
  char *res = REAL(__strndup)(src, n);
  __msan_copy_poison(res, src, copy_size);
  __msan_unpoison(res + copy_size, 1); // \0
  return res;
}

INTERCEPTOR(char *, gcvt, double number, SIZE_T ndigit, char *buf) {
  ENSURE_MSAN_INITED();
  char *res = REAL(gcvt)(number, ndigit, buf);
  // DynamoRio tool will take care of unpoisoning gcvt result for us.
  if (!__msan_has_dynamic_component()) {
    SIZE_T n = REAL(strlen)(buf);
    __msan_unpoison(buf, n + 1);
  }
  return res;
}

INTERCEPTOR(char *, strcat, char *dest, const char *src) {  // NOLINT
  ENSURE_MSAN_INITED();
  SIZE_T src_size = REAL(strlen)(src);
  SIZE_T dest_size = REAL(strlen)(dest);
  char *res = REAL(strcat)(dest, src);  // NOLINT
  __msan_copy_poison(dest + dest_size, src, src_size + 1);
  return res;
}

INTERCEPTOR(char *, strncat, char *dest, const char *src, SIZE_T n) {  // NOLINT
  ENSURE_MSAN_INITED();
  SIZE_T dest_size = REAL(strlen)(dest);
  SIZE_T copy_size = REAL(strlen)(src);
  if (copy_size < n)
    copy_size++;  // trailing \0
  char *res = REAL(strncat)(dest, src, n);  // NOLINT
  __msan_copy_poison(dest + dest_size, src, copy_size);
  return res;
}

INTERCEPTOR(long, strtol, const char *nptr, char **endptr,  // NOLINT
            int base) {
  ENSURE_MSAN_INITED();
  long res = REAL(strtol)(nptr, endptr, base);  // NOLINT
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(endptr, sizeof(*endptr));
  }
  return res;
}

INTERCEPTOR(long long, strtoll, const char *nptr, char **endptr,  // NOLINT
            int base) {
  ENSURE_MSAN_INITED();
  long res = REAL(strtoll)(nptr, endptr, base);  //NOLINT
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(endptr, sizeof(*endptr));
  }
  return res;
}

INTERCEPTOR(unsigned long, strtoul, const char *nptr, char **endptr,  // NOLINT
            int base) {
  ENSURE_MSAN_INITED();
  unsigned long res = REAL(strtoul)(nptr, endptr, base);  // NOLINT
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(endptr, sizeof(*endptr));
  }
  return res;
}

INTERCEPTOR(unsigned long long, strtoull, const char *nptr,  // NOLINT
            char **endptr, int base) {
  ENSURE_MSAN_INITED();
  unsigned long res = REAL(strtoull)(nptr, endptr, base);  // NOLINT
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(endptr, sizeof(*endptr));
  }
  return res;
}

INTERCEPTOR(double, strtod, const char *nptr, char **endptr) {  // NOLINT
  ENSURE_MSAN_INITED();
  double res = REAL(strtod)(nptr, endptr);  // NOLINT
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(endptr, sizeof(*endptr));
  }
  return res;
}

INTERCEPTOR(float, strtof, const char *nptr, char **endptr) {  // NOLINT
  ENSURE_MSAN_INITED();
  float res = REAL(strtof)(nptr, endptr);  // NOLINT
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(endptr, sizeof(*endptr));
  }
  return res;
}

INTERCEPTOR(long double, strtold, const char *nptr, char **endptr) {  // NOLINT
  ENSURE_MSAN_INITED();
  long double res = REAL(strtold)(nptr, endptr);  // NOLINT
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(endptr, sizeof(*endptr));
  }
  return res;
}

INTERCEPTOR(int, vasprintf, char **strp, const char *format, va_list ap) {
  ENSURE_MSAN_INITED();
  int res = REAL(vasprintf)(strp, format, ap);
  if (res >= 0 && !__msan_has_dynamic_component()) {
    __msan_unpoison(strp, sizeof(*strp));
    __msan_unpoison(*strp, res + 1);
  }
  return res;
}

INTERCEPTOR(int, asprintf, char **strp, const char *format, ...) {  // NOLINT
  ENSURE_MSAN_INITED();
  va_list ap;
  va_start(ap, format);
  int res = vasprintf(strp, format, ap);  // NOLINT
  va_end(ap);
  return res;
}

INTERCEPTOR(int, vsnprintf, char *str, uptr size,
            const char *format, va_list ap) {
  ENSURE_MSAN_INITED();
  int res = REAL(vsnprintf)(str, size, format, ap);
  if (res >= 0 && !__msan_has_dynamic_component()) {
    __msan_unpoison(str, res + 1);
  }
  return res;
}

INTERCEPTOR(int, vsprintf, char *str, const char *format, va_list ap) {
  ENSURE_MSAN_INITED();
  int res = REAL(vsprintf)(str, format, ap);
  if (res >= 0 && !__msan_has_dynamic_component()) {
    __msan_unpoison(str, res + 1);
  }
  return res;
}

INTERCEPTOR(int, vswprintf, void *str, uptr size, void *format, va_list ap) {
  ENSURE_MSAN_INITED();
  int res = REAL(vswprintf)(str, size, format, ap);
  if (res >= 0 && !__msan_has_dynamic_component()) {
    __msan_unpoison(str, 4 * (res + 1));
  }
  return res;
}

INTERCEPTOR(int, sprintf, char *str, const char *format, ...) {  // NOLINT
  ENSURE_MSAN_INITED();
  va_list ap;
  va_start(ap, format);
  int res = vsprintf(str, format, ap);  // NOLINT
  va_end(ap);
  return res;
}

INTERCEPTOR(int, snprintf, char *str, uptr size, const char *format, ...) {
  ENSURE_MSAN_INITED();
  va_list ap;
  va_start(ap, format);
  int res = vsnprintf(str, size, format, ap);
  va_end(ap);
  return res;
}

INTERCEPTOR(int, swprintf, void *str, uptr size, void *format, ...) {
  ENSURE_MSAN_INITED();
  va_list ap;
  va_start(ap, format);
  int res = vswprintf(str, size, format, ap);
  va_end(ap);
  return res;
}

// SIZE_T strftime(char *s, SIZE_T max, const char *format,const struct tm *tm);
INTERCEPTOR(SIZE_T, strftime, char *s, SIZE_T max, const char *format,
            void *tm) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(strftime)(s, max, format, tm);
  if (res) __msan_unpoison(s, res + 1);
  return res;
}

INTERCEPTOR(int, mbtowc, wchar_t *dest, const char *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  int res = REAL(mbtowc)(dest, src, n);
  if (res != -1 && dest) __msan_unpoison(dest, sizeof(wchar_t));
  return res;
}

INTERCEPTOR(int, mbrtowc, wchar_t *dest, const char *src, SIZE_T n, void *ps) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(mbrtowc)(dest, src, n, ps);
  if (res != (SIZE_T)-1 && dest) __msan_unpoison(dest, sizeof(wchar_t));
  return res;
}

INTERCEPTOR(SIZE_T, wcslen, const wchar_t *s) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(wcslen)(s);
  CHECK_UNPOISONED(s, sizeof(wchar_t) * (res + 1));
  return res;
}

// wchar_t *wcschr(const wchar_t *wcs, wchar_t wc);
INTERCEPTOR(wchar_t *, wcschr, void *s, wchar_t wc, void *ps) {
  ENSURE_MSAN_INITED();
  wchar_t *res = REAL(wcschr)(s, wc, ps);
  return res;
}

// wchar_t *wcscpy(wchar_t *dest, const wchar_t *src);
INTERCEPTOR(wchar_t *, wcscpy, wchar_t *dest, const wchar_t *src) {
  ENSURE_MSAN_INITED();
  wchar_t *res = REAL(wcscpy)(dest, src);
  __msan_copy_poison(dest, src, sizeof(wchar_t) * (REAL(wcslen)(src) + 1));
  return res;
}

// wchar_t *wmemcpy(wchar_t *dest, const wchar_t *src, SIZE_T n);
INTERCEPTOR(wchar_t *, wmemcpy, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  wchar_t *res = REAL(wmemcpy)(dest, src, n);
  __msan_copy_poison(dest, src, n * sizeof(wchar_t));
  return res;
}

INTERCEPTOR(wchar_t *, wmempcpy, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  wchar_t *res = REAL(wmempcpy)(dest, src, n);
  __msan_copy_poison(dest, src, n * sizeof(wchar_t));
  return res;
}

INTERCEPTOR(wchar_t *, wmemset, wchar_t *s, wchar_t c, SIZE_T n) {
  CHECK(MEM_IS_APP(s));
  ENSURE_MSAN_INITED();
  wchar_t *res = (wchar_t *)fast_memset(s, c, n * sizeof(wchar_t));
  __msan_unpoison(s, n * sizeof(wchar_t));
  return res;
}

INTERCEPTOR(wchar_t *, wmemmove, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  wchar_t *res = REAL(wmemmove)(dest, src, n);
  __msan_move_poison(dest, src, n * sizeof(wchar_t));
  return res;
}

INTERCEPTOR(int, wcscmp, const wchar_t *s1, const wchar_t *s2) {
  ENSURE_MSAN_INITED();
  int res = REAL(wcscmp)(s1, s2);
  return res;
}

INTERCEPTOR(double, wcstod, const wchar_t *nptr, wchar_t **endptr) {
  ENSURE_MSAN_INITED();
  double res = REAL(wcstod)(nptr, endptr);
  __msan_unpoison(endptr, sizeof(*endptr));
  return res;
}

// #define UNSUPPORTED(name) \
//   INTERCEPTOR(void, name, void) {                     \
//     Printf("MSAN: Unsupported %s\n", __FUNCTION__);   \
//     Die();                                            \
//   }

// FIXME: intercept the following functions:
// Note, they only matter when running without a dynamic tool.
// UNSUPPORTED(wcscoll_l)
// UNSUPPORTED(wcsnrtombs)
// UNSUPPORTED(wcstol)
// UNSUPPORTED(wcstoll)
// UNSUPPORTED(wcstold)
// UNSUPPORTED(wcstoul)
// UNSUPPORTED(wcstoull)
// UNSUPPORTED(wcsxfrm_l)
// UNSUPPORTED(wcsdup)
// UNSUPPORTED(wcsftime)
// UNSUPPORTED(wcsstr)
// UNSUPPORTED(wcsrchr)
// UNSUPPORTED(wctob)

INTERCEPTOR(int, gettimeofday, void *tv, void *tz) {
  ENSURE_MSAN_INITED();
  int res = REAL(gettimeofday)(tv, tz);
  if (tv)
    __msan_unpoison(tv, 16);
  if (tz)
    __msan_unpoison(tz, 8);
  return res;
}

INTERCEPTOR(char *, fcvt, double x, int a, int *b, int *c) {
  ENSURE_MSAN_INITED();
  char *res = REAL(fcvt)(x, a, b, c);
  if (!__msan_has_dynamic_component()) {
    __msan_unpoison(b, sizeof(*b));
    __msan_unpoison(c, sizeof(*c));
  }
  return res;
}

INTERCEPTOR(char *, getenv, char *name) {
  ENSURE_MSAN_INITED();
  char *res = REAL(getenv)(name);
  if (!__msan_has_dynamic_component()) {
    if (res)
      __msan_unpoison(res, REAL(strlen)(res) + 1);
  }
  return res;
}

extern char **environ;

static void UnpoisonEnviron() {
  char **envp = environ;
  for (; *envp; ++envp) {
    __msan_unpoison(envp, sizeof(*envp));
    __msan_unpoison(*envp, REAL(strlen)(*envp) + 1);
  }
  // Trailing NULL pointer.
  __msan_unpoison(envp, sizeof(*envp));
}

INTERCEPTOR(int, setenv, const char *name, const char *value, int overwrite) {
  ENSURE_MSAN_INITED();
  int res = REAL(setenv)(name, value, overwrite);
  if (!res) UnpoisonEnviron();
  return res;
}

INTERCEPTOR(int, putenv, char *string) {
  ENSURE_MSAN_INITED();
  int res = REAL(putenv)(string);
  if (!res) UnpoisonEnviron();
  return res;
}

INTERCEPTOR(int, __fxstat, int magic, int fd, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstat)(magic, fd, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat_sz);
  return res;
}

INTERCEPTOR(int, __fxstat64, int magic, int fd, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstat64)(magic, fd, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat64_sz);
  return res;
}

INTERCEPTOR(int, __xstat, int magic, char *path, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__xstat)(magic, path, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat_sz);
  return res;
}

INTERCEPTOR(int, __xstat64, int magic, char *path, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__xstat64)(magic, path, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat64_sz);
  return res;
}

INTERCEPTOR(int, __lxstat, int magic, char *path, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__lxstat)(magic, path, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat_sz);
  return res;
}

INTERCEPTOR(int, __lxstat64, int magic, char *path, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__lxstat64)(magic, path, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat64_sz);
  return res;
}

INTERCEPTOR(int, pipe, int pipefd[2]) {
  if (msan_init_is_running)
    return REAL(pipe)(pipefd);
  ENSURE_MSAN_INITED();
  int res = REAL(pipe)(pipefd);
  if (!res)
    __msan_unpoison(pipefd, sizeof(int[2]));
  return res;
}

INTERCEPTOR(int, pipe2, int pipefd[2], int flags) {
  ENSURE_MSAN_INITED();
  int res = REAL(pipe2)(pipefd, flags);
  if (!res)
    __msan_unpoison(pipefd, sizeof(int[2]));
  return res;
}

INTERCEPTOR(int, socketpair, int domain, int type, int protocol, int sv[2]) {
  ENSURE_MSAN_INITED();
  int res = REAL(socketpair)(domain, type, protocol, sv);
  if (!res)
    __msan_unpoison(sv, sizeof(int[2]));
  return res;
}

INTERCEPTOR(char *, fgets, char *s, int size, void *stream) {
  ENSURE_MSAN_INITED();
  char *res = REAL(fgets)(s, size, stream);
  if (res)
    __msan_unpoison(s, REAL(strlen)(s) + 1);
  return res;
}

INTERCEPTOR(char *, fgets_unlocked, char *s, int size, void *stream) {
  ENSURE_MSAN_INITED();
  char *res = REAL(fgets_unlocked)(s, size, stream);
  if (res)
    __msan_unpoison(s, REAL(strlen)(s) + 1);
  return res;
}

INTERCEPTOR(int, getrlimit, int resource, void *rlim) {
  if (msan_init_is_running)
    return REAL(getrlimit)(resource, rlim);
  ENSURE_MSAN_INITED();
  int res = REAL(getrlimit)(resource, rlim);
  if (!res)
    __msan_unpoison(rlim, __sanitizer::struct_rlimit_sz);
  return res;
}

INTERCEPTOR(int, getrlimit64, int resource, void *rlim) {
  if (msan_init_is_running)
    return REAL(getrlimit64)(resource, rlim);
  ENSURE_MSAN_INITED();
  int res = REAL(getrlimit64)(resource, rlim);
  if (!res)
    __msan_unpoison(rlim, __sanitizer::struct_rlimit64_sz);
  return res;
}

INTERCEPTOR(int, statfs, const char *s, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(statfs)(s, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_statfs_sz);
  return res;
}

INTERCEPTOR(int, fstatfs, int fd, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(fstatfs)(fd, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_statfs_sz);
  return res;
}

INTERCEPTOR(int, statfs64, const char *s, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(statfs64)(s, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_statfs64_sz);
  return res;
}

INTERCEPTOR(int, fstatfs64, int fd, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(fstatfs64)(fd, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_statfs64_sz);
  return res;
}

INTERCEPTOR(int, uname, void *utsname) {
  ENSURE_MSAN_INITED();
  int res = REAL(uname)(utsname);
  if (!res) {
    __msan_unpoison(utsname, __sanitizer::struct_utsname_sz);
  }
  return res;
}

INTERCEPTOR(int, gethostname, char *name, SIZE_T len) {
  ENSURE_MSAN_INITED();
  int res = REAL(gethostname)(name, len);
  if (!res) {
    SIZE_T real_len = REAL(strnlen)(name, len);
    if (real_len < len)
      ++real_len;
    __msan_unpoison(name, real_len);
  }
  return res;
}

INTERCEPTOR(int, epoll_wait, int epfd, void *events, int maxevents,
    int timeout) {
  ENSURE_MSAN_INITED();
  int res = REAL(epoll_wait)(epfd, events, maxevents, timeout);
  if (res > 0) {
    __msan_unpoison(events, __sanitizer::struct_epoll_event_sz * res);
  }
  return res;
}

INTERCEPTOR(int, epoll_pwait, int epfd, void *events, int maxevents,
    int timeout, void *sigmask) {
  ENSURE_MSAN_INITED();
  int res = REAL(epoll_pwait)(epfd, events, maxevents, timeout, sigmask);
  if (res > 0) {
    __msan_unpoison(events, __sanitizer::struct_epoll_event_sz * res);
  }
  return res;
}

INTERCEPTOR(SSIZE_T, recv, int fd, void *buf, SIZE_T len, int flags) {
  ENSURE_MSAN_INITED();
  SSIZE_T res = REAL(recv)(fd, buf, len, flags);
  if (res > 0)
    __msan_unpoison(buf, res);
  return res;
}

INTERCEPTOR(SSIZE_T, recvfrom, int fd, void *buf, SIZE_T len, int flags,
            void *srcaddr, int *addrlen) {
  ENSURE_MSAN_INITED();
  SIZE_T srcaddr_sz;
  if (srcaddr) srcaddr_sz = *addrlen;
  SSIZE_T res = REAL(recvfrom)(fd, buf, len, flags, srcaddr, addrlen);
  if (res > 0) {
    __msan_unpoison(buf, res);
    if (srcaddr) {
      SIZE_T sz = *addrlen;
      __msan_unpoison(srcaddr, (sz < srcaddr_sz) ? sz : srcaddr_sz);
    }
  }
  return res;
}

INTERCEPTOR(void *, calloc, SIZE_T nmemb, SIZE_T size) {
  if (CallocShouldReturnNullDueToOverflow(size, nmemb)) return 0;
  GET_MALLOC_STACK_TRACE;
  if (!msan_inited) {
    // Hack: dlsym calls calloc before REAL(calloc) is retrieved from dlsym.
    const SIZE_T kCallocPoolSize = 1024;
    static uptr calloc_memory_for_dlsym[kCallocPoolSize];
    static SIZE_T allocated;
    SIZE_T size_in_words = ((nmemb * size) + kWordSize - 1) / kWordSize;
    void *mem = (void*)&calloc_memory_for_dlsym[allocated];
    allocated += size_in_words;
    CHECK(allocated < kCallocPoolSize);
    return mem;
  }
  return MsanReallocate(&stack, 0, nmemb * size, sizeof(u64), true);
}

INTERCEPTOR(void *, realloc, void *ptr, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return MsanReallocate(&stack, ptr, size, sizeof(u64), false);
}

INTERCEPTOR(void *, malloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return MsanReallocate(&stack, 0, size, sizeof(u64), false);
}

void __msan_allocated_memory(const void* data, uptr size) {
  GET_MALLOC_STACK_TRACE;
  if (flags()->poison_in_malloc)
    __msan_poison(data, size);
  if (__msan_get_track_origins()) {
    u32 stack_id = StackDepotPut(stack.trace, stack.size);
    CHECK(stack_id);
    CHECK_EQ((stack_id >> 31), 0);  // Higher bit is occupied by stack origins.
    __msan_set_origin(data, size, stack_id);
  }
}

INTERCEPTOR(void *, mmap, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF_T offset) {
  ENSURE_MSAN_INITED();
  void *res = REAL(mmap)(addr, length, prot, flags, fd, offset);
  if (res != (void*)-1)
    __msan_unpoison(res, RoundUpTo(length, GetPageSize()));
  return res;
}

INTERCEPTOR(void *, mmap64, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF64_T offset) {
  ENSURE_MSAN_INITED();
  void *res = REAL(mmap64)(addr, length, prot, flags, fd, offset);
  if (res != (void*)-1)
    __msan_unpoison(res, RoundUpTo(length, GetPageSize()));
  return res;
}

struct dlinfo {
  char *dli_fname;
  void *dli_fbase;
  char *dli_sname;
  void *dli_saddr;
};

INTERCEPTOR(int, dladdr, void *addr, dlinfo *info) {
  ENSURE_MSAN_INITED();
  int res = REAL(dladdr)(addr, info);
  if (res != 0) {
    __msan_unpoison(info, sizeof(*info));
    if (info->dli_fname)
      __msan_unpoison(info->dli_fname, REAL(strlen)(info->dli_fname) + 1);
    if (info->dli_sname)
      __msan_unpoison(info->dli_sname, REAL(strlen)(info->dli_sname) + 1);
  }
  return res;
}

// dlopen() ultimately calls mmap() down inside the loader, which generally
// doesn't participate in dynamic symbol resolution.  Therefore we won't
// intercept its calls to mmap, and we have to hook it here.  The loader
// initializes the module before returning, so without the dynamic component, we
// won't be able to clear the shadow before the initializers.  Fixing this would
// require putting our own initializer first to clear the shadow.
INTERCEPTOR(void *, dlopen, const char *filename, int flag) {
  ENSURE_MSAN_INITED();
  EnterLoader();
  link_map *map = (link_map *)REAL(dlopen)(filename, flag);
  ExitLoader();
  if (!__msan_has_dynamic_component() && map) {
    // If msandr didn't clear the shadow before the initializers ran, we do it
    // ourselves afterwards.
    ForEachMappedRegion(map, __msan_unpoison);
  }
  return (void *)map;
}

typedef int (*dl_iterate_phdr_cb)(__sanitizer_dl_phdr_info *info, SIZE_T size,
                                  void *data);
struct dl_iterate_phdr_data {
  dl_iterate_phdr_cb callback;
  void *data;
};

static int msan_dl_iterate_phdr_cb(__sanitizer_dl_phdr_info *info, SIZE_T size,
                                   void *data) {
  if (info) {
    __msan_unpoison(info, size);
    if (info->dlpi_name)
      __msan_unpoison(info->dlpi_name, REAL(strlen)(info->dlpi_name) + 1);
  }
  dl_iterate_phdr_data *cbdata = (dl_iterate_phdr_data *)data;
  UnpoisonParam(3);
  return cbdata->callback(info, size, cbdata->data);
}

INTERCEPTOR(int, dl_iterate_phdr, dl_iterate_phdr_cb callback, void *data) {
  ENSURE_MSAN_INITED();
  EnterLoader();
  dl_iterate_phdr_data cbdata;
  cbdata.callback = callback;
  cbdata.data = data;
  int res = REAL(dl_iterate_phdr)(msan_dl_iterate_phdr_cb, (void *)&cbdata);
  ExitLoader();
  return res;
}

INTERCEPTOR(int, getrusage, int who, void *usage) {
  ENSURE_MSAN_INITED();
  int res = REAL(getrusage)(who, usage);
  if (res == 0) {
    __msan_unpoison(usage, __sanitizer::struct_rusage_sz);
  }
  return res;
}

const int kMaxSignals = 1024;
static uptr sigactions[kMaxSignals];
static StaticSpinMutex sigactions_mu;

static void SignalHandler(int signo) {
  typedef void (*signal_cb)(int x);
  signal_cb cb = (signal_cb)sigactions[signo];
  cb(signo);
}

static void SignalAction(int signo, void *si, void *uc) {
  UnpoisonParam(3);
  __msan_unpoison(si, __sanitizer::struct_sigaction_sz);
  __msan_unpoison(uc, __sanitizer::ucontext_t_sz);

  typedef void (*sigaction_cb)(int, void *, void *);
  sigaction_cb cb = (sigaction_cb)sigactions[signo];
  cb(signo, si, uc);
}

INTERCEPTOR(int, sigaction, int signo, const __sanitizer_sigaction *act,
            __sanitizer_sigaction *oldact) {
  ENSURE_MSAN_INITED();
  // FIXME: check that *act is unpoisoned.
  // That requires intercepting all of sigemptyset, sigfillset, etc.
  int res;
  if (flags()->wrap_signals) {
    SpinMutexLock lock(&sigactions_mu);
    CHECK_LT(signo, kMaxSignals);
    uptr old_cb = sigactions[signo];
    __sanitizer_sigaction new_act;
    __sanitizer_sigaction *pnew_act = act ? &new_act : 0;
    if (act) {
      internal_memcpy(pnew_act, act, __sanitizer::struct_sigaction_sz);
      uptr cb = __sanitizer::__sanitizer_get_sigaction_sa_sigaction(pnew_act);
      uptr new_cb =
          __sanitizer::__sanitizer_get_sigaction_sa_siginfo(pnew_act) ?
          (uptr)SignalAction : (uptr)SignalHandler;
      if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
        sigactions[signo] = cb;
        __sanitizer::__sanitizer_set_sigaction_sa_sigaction(pnew_act, new_cb);
      }
    }
    res = REAL(sigaction)(signo, pnew_act, oldact);
    if (res == 0 && oldact) {
      uptr cb = __sanitizer::__sanitizer_get_sigaction_sa_sigaction(oldact);
      if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
        __sanitizer::__sanitizer_set_sigaction_sa_sigaction(oldact, old_cb);
      }
    }
  } else {
    res = REAL(sigaction)(signo, act, oldact);
  }

  if (res == 0 && oldact) {
    __msan_unpoison(oldact, __sanitizer::struct_sigaction_sz);
  }
  return res;
}

INTERCEPTOR(int, signal, int signo, uptr cb) {
  ENSURE_MSAN_INITED();
  if (flags()->wrap_signals) {
    CHECK_LT(signo, kMaxSignals);
    SpinMutexLock lock(&sigactions_mu);
    if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
      sigactions[signo] = cb;
      cb = (uptr) SignalHandler;
    }
    return REAL(signal)(signo, cb);
  } else {
    return REAL(signal)(signo, cb);
  }
}

extern "C" int pthread_attr_init(void *attr);
extern "C" int pthread_attr_destroy(void *attr);
extern "C" int pthread_attr_setstacksize(void *attr, uptr stacksize);
extern "C" int pthread_attr_getstack(void *attr, uptr *stack, uptr *stacksize);

INTERCEPTOR(int, pthread_create, void *th, void *attr, void *(*callback)(void*),
            void * param) {
  ENSURE_MSAN_INITED(); // for GetTlsSize()
  __sanitizer_pthread_attr_t myattr;
  if (attr == 0) {
    pthread_attr_init(&myattr);
    attr = &myattr;
  }

  AdjustStackSizeLinux(attr, flags()->verbosity);

  int res = REAL(pthread_create)(th, attr, callback, param);
  if (attr == &myattr)
    pthread_attr_destroy(&myattr);
  if (!res) {
    __msan_unpoison(th, __sanitizer::pthread_t_sz);
  }
  return res;
}

INTERCEPTOR(int, pthread_key_create, __sanitizer_pthread_key_t *key,
            void (*dtor)(void *value)) {
  ENSURE_MSAN_INITED();
  int res = REAL(pthread_key_create)(key, dtor);
  if (!res && key)
    __msan_unpoison(key, sizeof(*key));
  return res;
}

INTERCEPTOR(int, pthread_join, void *th, void **retval) {
  ENSURE_MSAN_INITED();
  int res = REAL(pthread_join)(th, retval);
  if (!res && retval)
    __msan_unpoison(retval, sizeof(*retval));
  return res;
}

struct MSanInterceptorContext {
  bool in_interceptor_scope;
};

// A version of CHECK_UNPOISED using a saved scope value. Used in common
// interceptors.
#define CHECK_UNPOISONED_CTX(ctx, x, n)                         \
  do {                                                          \
    if (!((MSanInterceptorContext *)ctx)->in_interceptor_scope) \
      CHECK_UNPOISONED_0(x, n);                                 \
  } while (0)

#define COMMON_INTERCEPTOR_UNPOISON_PARAM(ctx, count)  \
  UnpoisonParam(count)
#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  __msan_unpoison(ptr, size)
#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size) \
  CHECK_UNPOISONED_CTX(ctx, ptr, size)
#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)              \
  if (msan_init_is_running) return REAL(func)(__VA_ARGS__);   \
  MSanInterceptorContext msan_ctx = {IsInInterceptorScope()}; \
  ctx = (void *)&msan_ctx;                                    \
  InterceptorScope interceptor_scope;                         \
  ENSURE_MSAN_INITED();
#define COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd) \
  do {                                         \
  } while (false)
#define COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd) \
  do {                                         \
  } while (false)
#define COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, newfd) \
  do {                                                      \
  } while (false)
#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) \
  do {                                                \
  } while (false)  // FIXME
#include "sanitizer_common/sanitizer_common_interceptors.inc"

#define COMMON_SYSCALL_PRE_READ_RANGE(p, s) CHECK_UNPOISONED(p, s)
#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s)
#define COMMON_SYSCALL_POST_READ_RANGE(p, s)
#define COMMON_SYSCALL_POST_WRITE_RANGE(p, s) __msan_unpoison(p, s)
#include "sanitizer_common/sanitizer_common_syscalls.inc"

// static
void *fast_memset(void *ptr, int c, SIZE_T n) {
  // hack until we have a really fast internal_memset
  if (sizeof(uptr) == 8 &&
      (n % 8) == 0 &&
      ((uptr)ptr % 8) == 0 &&
      (c == 0 || c == -1)) {
    // Printf("memset %p %zd %x\n", ptr, n, c);
    uptr to_store = c ? -1L : 0L;
    uptr *p = (uptr*)ptr;
    for (SIZE_T i = 0; i < n / 8; i++)
      p[i] = to_store;
    return ptr;
  }
  return internal_memset(ptr, c, n);
}

// static
void *fast_memcpy(void *dst, const void *src, SIZE_T n) {
  // Same hack as in fast_memset above.
  if (sizeof(uptr) == 8 &&
      (n % 8) == 0 &&
      ((uptr)dst % 8) == 0 &&
      ((uptr)src % 8) == 0) {
    uptr *d = (uptr*)dst;
    uptr *s = (uptr*)src;
    for (SIZE_T i = 0; i < n / 8; i++)
      d[i] = s[i];
    return dst;
  }
  return internal_memcpy(dst, src, n);
}

// These interface functions reside here so that they can use
// fast_memset, etc.
void __msan_unpoison(const void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  fast_memset((void*)MEM_TO_SHADOW((uptr)a), 0, size);
}

void __msan_poison(const void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  fast_memset((void*)MEM_TO_SHADOW((uptr)a),
              __msan::flags()->poison_heap_with_zeroes ? 0 : -1, size);
}

void __msan_poison_stack(void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  fast_memset((void*)MEM_TO_SHADOW((uptr)a),
              __msan::flags()->poison_stack_with_zeroes ? 0 : -1, size);
}

void __msan_clear_and_unpoison(void *a, uptr size) {
  fast_memset(a, 0, size);
  fast_memset((void*)MEM_TO_SHADOW((uptr)a), 0, size);
}

void __msan_copy_origin(void *dst, const void *src, uptr size) {
  if (!__msan_get_track_origins()) return;
  if (!MEM_IS_APP(dst) || !MEM_IS_APP(src)) return;
  uptr d = MEM_TO_ORIGIN(dst);
  uptr s = MEM_TO_ORIGIN(src);
  uptr beg = d & ~3UL;  // align down.
  uptr end = (d + size + 3) & ~3UL;  // align up.
  s = s & ~3UL;  // align down.
  fast_memcpy((void*)beg, (void*)s, end - beg);
}

void __msan_copy_poison(void *dst, const void *src, uptr size) {
  if (!MEM_IS_APP(dst)) return;
  if (!MEM_IS_APP(src)) return;
  fast_memcpy((void*)MEM_TO_SHADOW((uptr)dst),
              (void*)MEM_TO_SHADOW((uptr)src), size);
  __msan_copy_origin(dst, src, size);
}

void __msan_move_poison(void *dst, const void *src, uptr size) {
  if (!MEM_IS_APP(dst)) return;
  if (!MEM_IS_APP(src)) return;
  internal_memmove((void*)MEM_TO_SHADOW((uptr)dst),
         (void*)MEM_TO_SHADOW((uptr)src), size);
  __msan_copy_origin(dst, src, size);
}

void *__msan_memcpy(void *dest, const void *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  void *res = fast_memcpy(dest, src, n);
  __msan_copy_poison(dest, src, n);
  return res;
}

void *__msan_memset(void *s, int c, SIZE_T n) {
  ENSURE_MSAN_INITED();
  void *res = fast_memset(s, c, n);
  __msan_unpoison(s, n);
  return res;
}

void *__msan_memmove(void *dest, const void *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  void *res = REAL(memmove)(dest, src, n);
  __msan_move_poison(dest, src, n);
  return res;
}

namespace __msan {
void InitializeInterceptors() {
  static int inited = 0;
  CHECK_EQ(inited, 0);
  SANITIZER_COMMON_INTERCEPTORS_INIT;

  INTERCEPT_FUNCTION(mmap);
  INTERCEPT_FUNCTION(mmap64);
  INTERCEPT_FUNCTION(posix_memalign);
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(free);
  INTERCEPT_FUNCTION(fread);
  INTERCEPT_FUNCTION(fread_unlocked);
  INTERCEPT_FUNCTION(readlink);
  INTERCEPT_FUNCTION(memcpy);
  INTERCEPT_FUNCTION(mempcpy);
  INTERCEPT_FUNCTION(memset);
  INTERCEPT_FUNCTION(memmove);
  INTERCEPT_FUNCTION(bcopy);
  INTERCEPT_FUNCTION(wmemset);
  INTERCEPT_FUNCTION(wmemcpy);
  INTERCEPT_FUNCTION(wmempcpy);
  INTERCEPT_FUNCTION(wmemmove);
  INTERCEPT_FUNCTION(strcpy);  // NOLINT
  INTERCEPT_FUNCTION(stpcpy);  // NOLINT
  INTERCEPT_FUNCTION(strdup);
  INTERCEPT_FUNCTION(__strdup);
  INTERCEPT_FUNCTION(strndup);
  INTERCEPT_FUNCTION(__strndup);
  INTERCEPT_FUNCTION(strncpy);  // NOLINT
  INTERCEPT_FUNCTION(strlen);
  INTERCEPT_FUNCTION(strnlen);
  INTERCEPT_FUNCTION(gcvt);
  INTERCEPT_FUNCTION(strcat);  // NOLINT
  INTERCEPT_FUNCTION(strncat);  // NOLINT
  INTERCEPT_FUNCTION(strtol);
  INTERCEPT_FUNCTION(strtoll);
  INTERCEPT_FUNCTION(strtoul);
  INTERCEPT_FUNCTION(strtoull);
  INTERCEPT_FUNCTION(strtod);
  INTERCEPT_FUNCTION(strtof);
  INTERCEPT_FUNCTION(strtold);
  INTERCEPT_FUNCTION(vasprintf);
  INTERCEPT_FUNCTION(asprintf);
  INTERCEPT_FUNCTION(vsprintf);
  INTERCEPT_FUNCTION(vsnprintf);
  INTERCEPT_FUNCTION(vswprintf);
  INTERCEPT_FUNCTION(sprintf);  // NOLINT
  INTERCEPT_FUNCTION(snprintf);
  INTERCEPT_FUNCTION(swprintf);
  INTERCEPT_FUNCTION(strftime);
  INTERCEPT_FUNCTION(mbtowc);
  INTERCEPT_FUNCTION(mbrtowc);
  INTERCEPT_FUNCTION(wcslen);
  INTERCEPT_FUNCTION(wcschr);
  INTERCEPT_FUNCTION(wcscpy);
  INTERCEPT_FUNCTION(wcscmp);
  INTERCEPT_FUNCTION(wcstod);
  INTERCEPT_FUNCTION(getenv);
  INTERCEPT_FUNCTION(setenv);
  INTERCEPT_FUNCTION(putenv);
  INTERCEPT_FUNCTION(gettimeofday);
  INTERCEPT_FUNCTION(fcvt);
  INTERCEPT_FUNCTION(__fxstat);
  INTERCEPT_FUNCTION(__xstat);
  INTERCEPT_FUNCTION(__lxstat);
  INTERCEPT_FUNCTION(__fxstat64);
  INTERCEPT_FUNCTION(__xstat64);
  INTERCEPT_FUNCTION(__lxstat64);
  INTERCEPT_FUNCTION(pipe);
  INTERCEPT_FUNCTION(pipe2);
  INTERCEPT_FUNCTION(socketpair);
  INTERCEPT_FUNCTION(fgets);
  INTERCEPT_FUNCTION(fgets_unlocked);
  INTERCEPT_FUNCTION(getrlimit);
  INTERCEPT_FUNCTION(getrlimit64);
  INTERCEPT_FUNCTION(statfs);
  INTERCEPT_FUNCTION(fstatfs);
  INTERCEPT_FUNCTION(statfs64);
  INTERCEPT_FUNCTION(fstatfs64);
  INTERCEPT_FUNCTION(uname);
  INTERCEPT_FUNCTION(gethostname);
  INTERCEPT_FUNCTION(epoll_wait);
  INTERCEPT_FUNCTION(epoll_pwait);
  INTERCEPT_FUNCTION(recv);
  INTERCEPT_FUNCTION(recvfrom);
  INTERCEPT_FUNCTION(dladdr);
  INTERCEPT_FUNCTION(dlopen);
  INTERCEPT_FUNCTION(dl_iterate_phdr);
  INTERCEPT_FUNCTION(getrusage);
  INTERCEPT_FUNCTION(sigaction);
  INTERCEPT_FUNCTION(signal);
  INTERCEPT_FUNCTION(pthread_create);
  INTERCEPT_FUNCTION(pthread_key_create);
  INTERCEPT_FUNCTION(pthread_join);
  inited = 1;
}
}  // namespace __msan
