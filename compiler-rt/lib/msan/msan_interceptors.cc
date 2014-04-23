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

#include "msan.h"
#include "msan_thread.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_interception.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_linux.h"

#include <stdarg.h>
// ACHTUNG! No other system header includes in this file.
// Ideally, we should get rid of stdarg.h as well.

using namespace __msan;

using __sanitizer::memory_order;
using __sanitizer::atomic_load;
using __sanitizer::atomic_store;
using __sanitizer::atomic_uintptr_t;

// True if this is a nested interceptor.
static THREADLOCAL int in_interceptor_scope;

extern "C" int *__errno_location(void);

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
      Printf("UMR in %s at offset %d inside [%p, +%d) \n", __func__,         \
             offset, x, n);                                                  \
      __msan::PrintWarningWithOrigin(pc, bp,                                 \
                                     __msan_get_origin((char *)x + offset)); \
      if (__msan::flags()->halt_on_error) {                                  \
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

INTERCEPTOR(void *, memccpy, void *dest, const void *src, int c, SIZE_T n) {
  ENSURE_MSAN_INITED();
  void *res = REAL(memccpy)(dest, src, c, n);
  CHECK(!res || (res >= dest && res <= (char *)dest + n));
  SIZE_T sz = res ? (char *)res - (char *)dest : n;
  CHECK_UNPOISONED(src, sz);
  __msan_unpoison(dest, sz);
  return res;
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

INTERCEPTOR(void *, memalign, SIZE_T boundary, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  CHECK_EQ(boundary & (boundary - 1), 0);
  void *ptr = MsanReallocate(&stack, 0, size, boundary, false);
  return ptr;
}

INTERCEPTOR(void *, __libc_memalign, uptr align, uptr s)
    ALIAS(WRAPPER_NAME(memalign));

INTERCEPTOR(void *, valloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  void *ptr = MsanReallocate(&stack, 0, size, GetPageSizeCached(), false);
  return ptr;
}

INTERCEPTOR(void *, pvalloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  uptr PageSize = GetPageSizeCached();
  size = RoundUpTo(size, PageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = PageSize;
  }
  void *ptr = MsanReallocate(&stack, 0, size, PageSize, false);
  return ptr;
}

INTERCEPTOR(void, free, void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (ptr == 0) return;
  MsanDeallocate(&stack, ptr);
}

INTERCEPTOR(void, cfree, void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (ptr == 0) return;
  MsanDeallocate(&stack, ptr);
}

INTERCEPTOR(uptr, malloc_usable_size, void *ptr) {
  return __msan_get_allocated_size(ptr);
}

// This function actually returns a struct by value, but we can't unpoison a
// temporary! The following is equivalent on all supported platforms, and we
// have a test to confirm that.
INTERCEPTOR(void, mallinfo, __sanitizer_mallinfo *sret) {
  REAL(memset)(sret, 0, sizeof(*sret));
  __msan_unpoison(sret, sizeof(*sret));
}

INTERCEPTOR(int, mallopt, int cmd, int value) {
  return -1;
}

INTERCEPTOR(void, malloc_stats, void) {
  // FIXME: implement, but don't call REAL(malloc_stats)!
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
  GET_STORE_STACK_TRACE;
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(strcpy)(dest, src);  // NOLINT
  CopyPoison(dest, src, n + 1, &stack);
  return res;
}

INTERCEPTOR(char *, strncpy, char *dest, const char *src, SIZE_T n) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T copy_size = REAL(strnlen)(src, n);
  if (copy_size < n)
    copy_size++;  // trailing \0
  char *res = REAL(strncpy)(dest, src, n);  // NOLINT
  CopyPoison(dest, src, copy_size, &stack);
  return res;
}

INTERCEPTOR(char *, stpcpy, char *dest, const char *src) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(stpcpy)(dest, src);  // NOLINT
  CopyPoison(dest, src, n + 1, &stack);
  return res;
}

INTERCEPTOR(char *, strdup, char *src) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(strdup)(src);
  CopyPoison(res, src, n + 1, &stack);
  return res;
}

INTERCEPTOR(char *, __strdup, char *src) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T n = REAL(strlen)(src);
  char *res = REAL(__strdup)(src);
  CopyPoison(res, src, n + 1, &stack);
  return res;
}

INTERCEPTOR(char *, strndup, char *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T copy_size = REAL(strnlen)(src, n);
  char *res = REAL(strndup)(src, n);
  CopyPoison(res, src, copy_size, &stack);
  __msan_unpoison(res + copy_size, 1); // \0
  return res;
}

INTERCEPTOR(char *, __strndup, char *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T copy_size = REAL(strnlen)(src, n);
  char *res = REAL(__strndup)(src, n);
  CopyPoison(res, src, copy_size, &stack);
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
  GET_STORE_STACK_TRACE;
  SIZE_T src_size = REAL(strlen)(src);
  SIZE_T dest_size = REAL(strlen)(dest);
  char *res = REAL(strcat)(dest, src);  // NOLINT
  CopyPoison(dest + dest_size, src, src_size + 1, &stack);
  return res;
}

INTERCEPTOR(char *, strncat, char *dest, const char *src, SIZE_T n) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T dest_size = REAL(strlen)(dest);
  SIZE_T copy_size = REAL(strnlen)(src, n);
  char *res = REAL(strncat)(dest, src, n);  // NOLINT
  CopyPoison(dest + dest_size, src, copy_size, &stack);
  __msan_unpoison(dest + dest_size + copy_size, 1); // \0
  return res;
}

// Hack: always pass nptr and endptr as part of __VA_ARGS_ to avoid having to
// deal with empty __VA_ARGS__ in the case of INTERCEPTOR_STRTO.
#define INTERCEPTOR_STRTO_BODY(ret_type, func, ...) \
  ENSURE_MSAN_INITED();                             \
  ret_type res = REAL(func)(__VA_ARGS__);           \
  if (!__msan_has_dynamic_component()) {            \
    __msan_unpoison(endptr, sizeof(*endptr));       \
  }                                                 \
  return res;

#define INTERCEPTOR_STRTO(ret_type, func)                        \
  INTERCEPTOR(ret_type, func, const char *nptr, char **endptr) { \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr);        \
  }

#define INTERCEPTOR_STRTO_BASE(ret_type, func)                             \
  INTERCEPTOR(ret_type, func, const char *nptr, char **endptr, int base) { \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr, base);            \
  }

#define INTERCEPTOR_STRTO_LOC(ret_type, func)                               \
  INTERCEPTOR(ret_type, func, const char *nptr, char **endptr, void *loc) { \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr, loc);              \
  }

#define INTERCEPTOR_STRTO_BASE_LOC(ret_type, func)                       \
  INTERCEPTOR(ret_type, func, const char *nptr, char **endptr, int base, \
              void *loc) {                                               \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr, base, loc);     \
  }

INTERCEPTOR_STRTO(double, strtod)                           // NOLINT
INTERCEPTOR_STRTO(float, strtof)                            // NOLINT
INTERCEPTOR_STRTO(long double, strtold)                     // NOLINT
INTERCEPTOR_STRTO_BASE(long, strtol)                        // NOLINT
INTERCEPTOR_STRTO_BASE(long long, strtoll)                  // NOLINT
INTERCEPTOR_STRTO_BASE(unsigned long, strtoul)              // NOLINT
INTERCEPTOR_STRTO_BASE(unsigned long long, strtoull)        // NOLINT
INTERCEPTOR_STRTO_LOC(double, strtod_l)                     // NOLINT
INTERCEPTOR_STRTO_LOC(double, __strtod_l)                   // NOLINT
INTERCEPTOR_STRTO_LOC(float, strtof_l)                      // NOLINT
INTERCEPTOR_STRTO_LOC(float, __strtof_l)                    // NOLINT
INTERCEPTOR_STRTO_LOC(long double, strtold_l)               // NOLINT
INTERCEPTOR_STRTO_LOC(long double, __strtold_l)             // NOLINT
INTERCEPTOR_STRTO_BASE_LOC(long, strtol_l)                  // NOLINT
INTERCEPTOR_STRTO_BASE_LOC(long long, strtoll_l)            // NOLINT
INTERCEPTOR_STRTO_BASE_LOC(unsigned long, strtoul_l)        // NOLINT
INTERCEPTOR_STRTO_BASE_LOC(unsigned long long, strtoull_l)  // NOLINT

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

#define INTERCEPTOR_STRFTIME_BODY(char_type, ret_type, func, s, ...) \
  ENSURE_MSAN_INITED();                                              \
  ret_type res = REAL(func)(s, __VA_ARGS__);                         \
  if (s) __msan_unpoison(s, sizeof(char_type) * (res + 1));          \
  return res;

INTERCEPTOR(SIZE_T, strftime, char *s, SIZE_T max, const char *format,
            __sanitizer_tm *tm) {
  INTERCEPTOR_STRFTIME_BODY(char, SIZE_T, strftime, s, max, format, tm);
}

INTERCEPTOR(SIZE_T, strftime_l, char *s, SIZE_T max, const char *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(char, SIZE_T, strftime_l, s, max, format, tm, loc);
}

INTERCEPTOR(SIZE_T, __strftime_l, char *s, SIZE_T max, const char *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(char, SIZE_T, __strftime_l, s, max, format, tm,
                            loc);
}

INTERCEPTOR(SIZE_T, wcsftime, wchar_t *s, SIZE_T max, const wchar_t *format,
            __sanitizer_tm *tm) {
  INTERCEPTOR_STRFTIME_BODY(wchar_t, SIZE_T, wcsftime, s, max, format, tm);
}

INTERCEPTOR(SIZE_T, wcsftime_l, wchar_t *s, SIZE_T max, const wchar_t *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(wchar_t, SIZE_T, wcsftime_l, s, max, format, tm,
                            loc);
}

INTERCEPTOR(SIZE_T, __wcsftime_l, wchar_t *s, SIZE_T max, const wchar_t *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(wchar_t, SIZE_T, __wcsftime_l, s, max, format, tm,
                            loc);
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
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wcscpy)(dest, src);
  CopyPoison(dest, src, sizeof(wchar_t) * (REAL(wcslen)(src) + 1), &stack);
  return res;
}

// wchar_t *wmemcpy(wchar_t *dest, const wchar_t *src, SIZE_T n);
INTERCEPTOR(wchar_t *, wmemcpy, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wmemcpy)(dest, src, n);
  CopyPoison(dest, src, n * sizeof(wchar_t), &stack);
  return res;
}

INTERCEPTOR(wchar_t *, wmempcpy, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wmempcpy)(dest, src, n);
  CopyPoison(dest, src, n * sizeof(wchar_t), &stack);
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
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wmemmove)(dest, src, n);
  MovePoison(dest, src, n * sizeof(wchar_t), &stack);
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

INTERCEPTOR(int, __fxstatat, int magic, int fd, char *pathname, void *buf,
            int flags) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstatat)(magic, fd, pathname, buf, flags);
  if (!res) __msan_unpoison(buf, __sanitizer::struct_stat_sz);
  return res;
}

INTERCEPTOR(int, __fxstatat64, int magic, int fd, char *pathname, void *buf,
            int flags) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstatat64)(magic, fd, pathname, buf, flags);
  if (!res) __msan_unpoison(buf, __sanitizer::struct_stat64_sz);
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
  if (CallocShouldReturnNullDueToOverflow(size, nmemb))
    return AllocatorReturnNull();
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
  if (addr && !MEM_IS_APP(addr)) {
    if (flags & map_fixed) {
      *__errno_location() = errno_EINVAL;
      return (void *)-1;
    } else {
      addr = 0;
    }
  }
  void *res = REAL(mmap)(addr, length, prot, flags, fd, offset);
  if (res != (void*)-1)
    __msan_unpoison(res, RoundUpTo(length, GetPageSize()));
  return res;
}

INTERCEPTOR(void *, mmap64, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF64_T offset) {
  ENSURE_MSAN_INITED();
  if (addr && !MEM_IS_APP(addr)) {
    if (flags & map_fixed) {
      *__errno_location() = errno_EINVAL;
      return (void *)-1;
    } else {
      addr = 0;
    }
  }
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

INTERCEPTOR(char *, dlerror, int fake) {
  ENSURE_MSAN_INITED();
  char *res = REAL(dlerror)(fake);
  if (res != 0) __msan_unpoison(res, REAL(strlen)(res) + 1);
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
  return IndirectExternCall(cbdata->callback)(info, size, cbdata->data);
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

class SignalHandlerScope {
 public:
  SignalHandlerScope() { GetCurrentThread()->EnterSignalHandler(); }
  ~SignalHandlerScope() { GetCurrentThread()->LeaveSignalHandler(); }
};

// sigactions_mu guarantees atomicity of sigaction() and signal() calls.
// Access to sigactions[] is gone with relaxed atomics to avoid data race with
// the signal handler.
const int kMaxSignals = 1024;
static atomic_uintptr_t sigactions[kMaxSignals];
static StaticSpinMutex sigactions_mu;

static void SignalHandler(int signo) {
  SignalHandlerScope signal_handler_scope;
  ScopedThreadLocalStateBackup stlsb;
  UnpoisonParam(1);

  typedef void (*signal_cb)(int x);
  signal_cb cb =
      (signal_cb)atomic_load(&sigactions[signo], memory_order_relaxed);
  IndirectExternCall(cb)(signo);
}

static void SignalAction(int signo, void *si, void *uc) {
  SignalHandlerScope signal_handler_scope;
  ScopedThreadLocalStateBackup stlsb;
  UnpoisonParam(3);
  __msan_unpoison(si, sizeof(__sanitizer_sigaction));
  __msan_unpoison(uc, __sanitizer::ucontext_t_sz);

  typedef void (*sigaction_cb)(int, void *, void *);
  sigaction_cb cb =
      (sigaction_cb)atomic_load(&sigactions[signo], memory_order_relaxed);
  IndirectExternCall(cb)(signo, si, uc);
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
    uptr old_cb = atomic_load(&sigactions[signo], memory_order_relaxed);
    __sanitizer_sigaction new_act;
    __sanitizer_sigaction *pnew_act = act ? &new_act : 0;
    if (act) {
      internal_memcpy(pnew_act, act, sizeof(__sanitizer_sigaction));
      uptr cb = (uptr)pnew_act->sigaction;
      uptr new_cb = (pnew_act->sa_flags & __sanitizer::sa_siginfo)
                        ? (uptr)SignalAction
                        : (uptr)SignalHandler;
      if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
        atomic_store(&sigactions[signo], cb, memory_order_relaxed);
        pnew_act->sigaction = (void (*)(int, void *, void *))new_cb;
      }
    }
    res = REAL(sigaction)(signo, pnew_act, oldact);
    if (res == 0 && oldact) {
      uptr cb = (uptr)oldact->sigaction;
      if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
        oldact->sigaction = (void (*)(int, void *, void *))old_cb;
      }
    }
  } else {
    res = REAL(sigaction)(signo, act, oldact);
  }

  if (res == 0 && oldact) {
    __msan_unpoison(oldact, sizeof(__sanitizer_sigaction));
  }
  return res;
}

INTERCEPTOR(int, signal, int signo, uptr cb) {
  ENSURE_MSAN_INITED();
  if (flags()->wrap_signals) {
    CHECK_LT(signo, kMaxSignals);
    SpinMutexLock lock(&sigactions_mu);
    if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
      atomic_store(&sigactions[signo], cb, memory_order_relaxed);
      cb = (uptr) SignalHandler;
    }
    return REAL(signal)(signo, cb);
  } else {
    return REAL(signal)(signo, cb);
  }
}

extern "C" int pthread_attr_init(void *attr);
extern "C" int pthread_attr_destroy(void *attr);

static void *MsanThreadStartFunc(void *arg) {
  MsanThread *t = (MsanThread *)arg;
  SetCurrentThread(t);
  return t->ThreadStart();
}

INTERCEPTOR(int, pthread_create, void *th, void *attr, void *(*callback)(void*),
            void * param) {
  ENSURE_MSAN_INITED(); // for GetTlsSize()
  __sanitizer_pthread_attr_t myattr;
  if (attr == 0) {
    pthread_attr_init(&myattr);
    attr = &myattr;
  }

  AdjustStackSize(attr);

  MsanThread *t = MsanThread::Create(callback, param);

  int res = REAL(pthread_create)(th, attr, MsanThreadStartFunc, t);

  if (attr == &myattr)
    pthread_attr_destroy(&myattr);
  if (!res) {
    __msan_unpoison(th, __sanitizer::pthread_t_sz);
  }
  return res;
}

INTERCEPTOR(int, pthread_key_create, __sanitizer_pthread_key_t *key,
            void (*dtor)(void *value)) {
  if (msan_init_is_running) return REAL(pthread_key_create)(key, dtor);
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

extern char *tzname[2];

INTERCEPTOR(void, tzset, int fake) {
  ENSURE_MSAN_INITED();
  REAL(tzset)(fake);
  if (tzname[0])
    __msan_unpoison(tzname[0], REAL(strlen)(tzname[0]) + 1);
  if (tzname[1])
    __msan_unpoison(tzname[1], REAL(strlen)(tzname[1]) + 1);
  return;
}

struct MSanAtExitRecord {
  void (*func)(void *arg);
  void *arg;
};

void MSanAtExitWrapper(void *arg) {
  UnpoisonParam(1);
  MSanAtExitRecord *r = (MSanAtExitRecord *)arg;
  IndirectExternCall(r->func)(r->arg);
  InternalFree(r);
}

// Unpoison argument shadow for C++ module destructors.
INTERCEPTOR(int, __cxa_atexit, void (*func)(void *), void *arg,
            void *dso_handle) {
  if (msan_init_is_running) return REAL(__cxa_atexit)(func, arg, dso_handle);
  ENSURE_MSAN_INITED();
  MSanAtExitRecord *r =
      (MSanAtExitRecord *)InternalAlloc(sizeof(MSanAtExitRecord));
  r->func = func;
  r->arg = arg;
  return REAL(__cxa_atexit)(MSanAtExitWrapper, r, dso_handle);
}

DECLARE_REAL(int, shmctl, int shmid, int cmd, void *buf)

INTERCEPTOR(void *, shmat, int shmid, const void *shmaddr, int shmflg) {
  ENSURE_MSAN_INITED();
  void *p = REAL(shmat)(shmid, shmaddr, shmflg);
  if (p != (void *)-1) {
    __sanitizer_shmid_ds ds;
    int res = REAL(shmctl)(shmid, shmctl_ipc_stat, &ds);
    if (!res) {
      __msan_unpoison(p, ds.shm_segsz);
    }
  }
  return p;
}

// Linux kernel has a bug that leads to kernel deadlock if a process
// maps TBs of memory and then calls mlock().
static void MlockIsUnsupported() {
  static atomic_uint8_t printed;
  if (atomic_exchange(&printed, 1, memory_order_relaxed))
    return;
  VPrintf(1,
          "INFO: MemorySanitizer ignores mlock/mlockall/munlock/munlockall\n");
}

INTERCEPTOR(int, mlock, const void *addr, uptr len) {
  MlockIsUnsupported();
  return 0;
}

INTERCEPTOR(int, munlock, const void *addr, uptr len) {
  MlockIsUnsupported();
  return 0;
}

INTERCEPTOR(int, mlockall, int flags) {
  MlockIsUnsupported();
  return 0;
}

INTERCEPTOR(int, munlockall, void) {
  MlockIsUnsupported();
  return 0;
}

struct MSanInterceptorContext {
  bool in_interceptor_scope;
};

namespace __msan {

int OnExit() {
  // FIXME: ask frontend whether we need to return failure.
  return 0;
}

}  // namespace __msan

// A version of CHECK_UNPOISONED using a saved scope value. Used in common
// interceptors.
#define CHECK_UNPOISONED_CTX(ctx, x, n)                         \
  do {                                                          \
    if (!((MSanInterceptorContext *)ctx)->in_interceptor_scope) \
      CHECK_UNPOISONED_0(x, n);                                 \
  } while (0)

#define MSAN_INTERCEPT_FUNC(name)                                       \
  do {                                                                  \
    if ((!INTERCEPT_FUNCTION(name) || !REAL(name)))                     \
      VReport(1, "MemorySanitizer: failed to intercept '" #name "'\n"); \
  } while (0)

#define COMMON_INTERCEPT_FUNCTION(name) MSAN_INTERCEPT_FUNC(name)
#define COMMON_INTERCEPTOR_UNPOISON_PARAM(count)  \
  UnpoisonParam(count)
#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  __msan_unpoison(ptr, size)
#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size) \
  CHECK_UNPOISONED_CTX(ctx, ptr, size)
#define COMMON_INTERCEPTOR_INITIALIZE_RANGE(ptr, size) \
  __msan_unpoison(ptr, size)
#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)                  \
  if (msan_init_is_running) return REAL(func)(__VA_ARGS__);       \
  MSanInterceptorContext msan_ctx = {IsInInterceptorScope()};     \
  ctx = (void *)&msan_ctx;                                        \
  (void)ctx;                                                      \
  InterceptorScope interceptor_scope;                             \
  __msan_unpoison(__errno_location(), sizeof(int)); /* NOLINT */  \
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
#define COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name) \
  do {                                                         \
  } while (false)  // FIXME
#define COMMON_INTERCEPTOR_BLOCK_REAL(name) REAL(name)
#define COMMON_INTERCEPTOR_ON_EXIT(ctx) OnExit()
// FIXME: update Msan to use common printf interceptors
#define SANITIZER_INTERCEPT_PRINTF 0
#include "sanitizer_common/sanitizer_common_interceptors.inc"

#define COMMON_SYSCALL_PRE_READ_RANGE(p, s) CHECK_UNPOISONED(p, s)
#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s) \
  do {                                       \
  } while (false)
#define COMMON_SYSCALL_POST_READ_RANGE(p, s) \
  do {                                       \
  } while (false)
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

static void PoisonShadow(uptr ptr, uptr size, u8 value) {
  uptr PageSize = GetPageSizeCached();
  uptr shadow_beg = MEM_TO_SHADOW(ptr);
  uptr shadow_end = MEM_TO_SHADOW(ptr + size);
  if (value ||
      shadow_end - shadow_beg < common_flags()->clear_shadow_mmap_threshold) {
    fast_memset((void*)shadow_beg, value, shadow_end - shadow_beg);
  } else {
    uptr page_beg = RoundUpTo(shadow_beg, PageSize);
    uptr page_end = RoundDownTo(shadow_end, PageSize);

    if (page_beg >= page_end) {
      fast_memset((void *)shadow_beg, 0, shadow_end - shadow_beg);
    } else {
      if (page_beg != shadow_beg) {
        fast_memset((void *)shadow_beg, 0, page_beg - shadow_beg);
      }
      if (page_end != shadow_end) {
        fast_memset((void *)page_end, 0, shadow_end - page_end);
      }
      MmapFixedNoReserve(page_beg, page_end - page_beg);
    }
  }
}

// These interface functions reside here so that they can use
// fast_memset, etc.
void __msan_unpoison(const void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  PoisonShadow((uptr)a, size, 0);
}

void __msan_poison(const void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  PoisonShadow((uptr)a, size,
               __msan::flags()->poison_heap_with_zeroes ? 0 : -1);
}

void __msan_poison_stack(void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  PoisonShadow((uptr)a, size,
               __msan::flags()->poison_stack_with_zeroes ? 0 : -1);
}

void __msan_clear_and_unpoison(void *a, uptr size) {
  fast_memset(a, 0, size);
  PoisonShadow((uptr)a, size, 0);
}

void *__msan_memcpy(void *dest, const void *src, SIZE_T n) {
  if (!msan_inited) return internal_memcpy(dest, src, n);
  if (msan_init_is_running) return REAL(memcpy)(dest, src, n);
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  void *res = fast_memcpy(dest, src, n);
  CopyPoison(dest, src, n, &stack);
  return res;
}

void *__msan_memset(void *s, int c, SIZE_T n) {
  if (!msan_inited) return internal_memset(s, c, n);
  if (msan_init_is_running) return REAL(memset)(s, c, n);
  ENSURE_MSAN_INITED();
  void *res = fast_memset(s, c, n);
  __msan_unpoison(s, n);
  return res;
}

void *__msan_memmove(void *dest, const void *src, SIZE_T n) {
  if (!msan_inited) return internal_memmove(dest, src, n);
  if (msan_init_is_running) return REAL(memmove)(dest, src, n);
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  void *res = REAL(memmove)(dest, src, n);
  MovePoison(dest, src, n, &stack);
  return res;
}

void __msan_unpoison_string(const char* s) {
  if (!MEM_IS_APP(s)) return;
  __msan_unpoison(s, REAL(strlen)(s) + 1);
}

namespace __msan {

u32 GetOriginIfPoisoned(uptr addr, uptr size) {
  unsigned char *s = (unsigned char *)MEM_TO_SHADOW(addr);
  for (uptr i = 0; i < size; ++i)
    if (s[i])
      return *(u32 *)SHADOW_TO_ORIGIN((s + i) & ~3UL);
  return 0;
}

void SetOriginIfPoisoned(uptr addr, uptr src_shadow, uptr size,
                         u32 src_origin) {
  uptr dst_s = MEM_TO_SHADOW(addr);
  uptr src_s = src_shadow;
  uptr src_s_end = src_s + size;

  for (; src_s < src_s_end; ++dst_s, ++src_s)
    if (*(u8 *)src_s) *(u32 *)SHADOW_TO_ORIGIN(dst_s &~3UL) = src_origin;
}

void CopyOrigin(void *dst, const void *src, uptr size, StackTrace *stack) {
  if (!__msan_get_track_origins()) return;
  if (!MEM_IS_APP(dst) || !MEM_IS_APP(src)) return;

  uptr d = (uptr)dst;
  uptr beg = d & ~3UL;
  // Copy left unaligned origin if that memory is poisoned.
  if (beg < d) {
    u32 o = GetOriginIfPoisoned(beg, d - beg);
    if (o) {
      if (__msan_get_track_origins() > 1) o = ChainOrigin(o, stack);
      *(u32 *)MEM_TO_ORIGIN(beg) = o;
    }
    beg += 4;
  }

  uptr end = (d + size + 3) & ~3UL;
  // Copy right unaligned origin if that memory is poisoned.
  if (end > d + size) {
    u32 o = GetOriginIfPoisoned(d + size, end - d - size);
    if (o) {
      if (__msan_get_track_origins() > 1) o = ChainOrigin(o, stack);
      *(u32 *)MEM_TO_ORIGIN(end - 4) = o;
    }
    end -= 4;
  }

  if (beg < end) {
    // Align src up.
    uptr s = ((uptr)src + 3) & ~3UL;
    // FIXME: factor out to msan_copy_origin_aligned
    if (__msan_get_track_origins() > 1) {
      u32 *src = (u32 *)MEM_TO_ORIGIN(s);
      u32 *src_s = (u32 *)MEM_TO_SHADOW(s);
      u32 *src_end = src + (end - beg);
      u32 *dst = (u32 *)MEM_TO_ORIGIN(beg);
      u32 src_o = 0;
      u32 dst_o = 0;
      for (; src < src_end; ++src, ++src_s, ++dst) {
        if (!*src_s) continue;
        if (*src != src_o) {
          src_o = *src;
          dst_o = ChainOrigin(src_o, stack);
        }
        *dst = dst_o;
      }
    } else {
      fast_memcpy((void *)MEM_TO_ORIGIN(beg), (void *)MEM_TO_ORIGIN(s),
                  end - beg);
    }
  }
}

void MovePoison(void *dst, const void *src, uptr size, StackTrace *stack) {
  if (!MEM_IS_APP(dst)) return;
  if (!MEM_IS_APP(src)) return;
  if (src == dst) return;
  internal_memmove((void *)MEM_TO_SHADOW((uptr)dst),
                   (void *)MEM_TO_SHADOW((uptr)src), size);
  CopyOrigin(dst, src, size, stack);
}

void CopyPoison(void *dst, const void *src, uptr size, StackTrace *stack) {
  if (!MEM_IS_APP(dst)) return;
  if (!MEM_IS_APP(src)) return;
  fast_memcpy((void *)MEM_TO_SHADOW((uptr)dst),
              (void *)MEM_TO_SHADOW((uptr)src), size);
  CopyOrigin(dst, src, size, stack);
}

void InitializeInterceptors() {
  static int inited = 0;
  CHECK_EQ(inited, 0);
  SANITIZER_COMMON_INTERCEPTORS_INIT;

  INTERCEPT_FUNCTION(mmap);
  INTERCEPT_FUNCTION(mmap64);
  INTERCEPT_FUNCTION(posix_memalign);
  INTERCEPT_FUNCTION(memalign);
  INTERCEPT_FUNCTION(valloc);
  INTERCEPT_FUNCTION(pvalloc);
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(free);
  INTERCEPT_FUNCTION(cfree);
  INTERCEPT_FUNCTION(malloc_usable_size);
  INTERCEPT_FUNCTION(mallinfo);
  INTERCEPT_FUNCTION(mallopt);
  INTERCEPT_FUNCTION(malloc_stats);
  INTERCEPT_FUNCTION(fread);
  INTERCEPT_FUNCTION(fread_unlocked);
  INTERCEPT_FUNCTION(readlink);
  INTERCEPT_FUNCTION(memcpy);
  INTERCEPT_FUNCTION(memccpy);
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
  INTERCEPT_FUNCTION(strtod_l);
  INTERCEPT_FUNCTION(__strtod_l);
  INTERCEPT_FUNCTION(strtof);
  INTERCEPT_FUNCTION(strtof_l);
  INTERCEPT_FUNCTION(__strtof_l);
  INTERCEPT_FUNCTION(strtold);
  INTERCEPT_FUNCTION(strtold_l);
  INTERCEPT_FUNCTION(__strtold_l);
  INTERCEPT_FUNCTION(strtol_l);
  INTERCEPT_FUNCTION(strtoll_l);
  INTERCEPT_FUNCTION(strtoul_l);
  INTERCEPT_FUNCTION(strtoull_l);
  INTERCEPT_FUNCTION(vasprintf);
  INTERCEPT_FUNCTION(asprintf);
  INTERCEPT_FUNCTION(vsprintf);
  INTERCEPT_FUNCTION(vsnprintf);
  INTERCEPT_FUNCTION(vswprintf);
  INTERCEPT_FUNCTION(sprintf);  // NOLINT
  INTERCEPT_FUNCTION(snprintf);
  INTERCEPT_FUNCTION(swprintf);
  INTERCEPT_FUNCTION(strftime);
  INTERCEPT_FUNCTION(strftime_l);
  INTERCEPT_FUNCTION(__strftime_l);
  INTERCEPT_FUNCTION(wcsftime);
  INTERCEPT_FUNCTION(wcsftime_l);
  INTERCEPT_FUNCTION(__wcsftime_l);
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
  INTERCEPT_FUNCTION(__fxstatat);
  INTERCEPT_FUNCTION(__xstat);
  INTERCEPT_FUNCTION(__lxstat);
  INTERCEPT_FUNCTION(__fxstat64);
  INTERCEPT_FUNCTION(__fxstatat64);
  INTERCEPT_FUNCTION(__xstat64);
  INTERCEPT_FUNCTION(__lxstat64);
  INTERCEPT_FUNCTION(pipe);
  INTERCEPT_FUNCTION(pipe2);
  INTERCEPT_FUNCTION(socketpair);
  INTERCEPT_FUNCTION(fgets);
  INTERCEPT_FUNCTION(fgets_unlocked);
  INTERCEPT_FUNCTION(getrlimit);
  INTERCEPT_FUNCTION(getrlimit64);
  INTERCEPT_FUNCTION(uname);
  INTERCEPT_FUNCTION(gethostname);
  INTERCEPT_FUNCTION(epoll_wait);
  INTERCEPT_FUNCTION(epoll_pwait);
  INTERCEPT_FUNCTION(recv);
  INTERCEPT_FUNCTION(recvfrom);
  INTERCEPT_FUNCTION(dladdr);
  INTERCEPT_FUNCTION(dlerror);
  INTERCEPT_FUNCTION(dlopen);
  INTERCEPT_FUNCTION(dl_iterate_phdr);
  INTERCEPT_FUNCTION(getrusage);
  INTERCEPT_FUNCTION(sigaction);
  INTERCEPT_FUNCTION(signal);
  INTERCEPT_FUNCTION(pthread_create);
  INTERCEPT_FUNCTION(pthread_key_create);
  INTERCEPT_FUNCTION(pthread_join);
  INTERCEPT_FUNCTION(tzset);
  INTERCEPT_FUNCTION(__cxa_atexit);
  INTERCEPT_FUNCTION(shmat);

  inited = 1;
}
}  // namespace __msan
