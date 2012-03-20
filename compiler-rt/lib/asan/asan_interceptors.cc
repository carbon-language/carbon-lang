//===-- asan_interceptors.cc ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Intercept various libc functions.
//===----------------------------------------------------------------------===//
#include "asan_interceptors.h"

#include "asan_allocator.h"
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_mac.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread_registry.h"
#include "interception/interception.h"

#include <new>

#if defined(__APPLE__)
// FIXME(samsonov): Gradually replace system headers with declarations of
// intercepted functions.
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <strings.h>
#endif  // __APPLE__

#if defined(__APPLE__)
extern "C" {
void longjmp(void* env, int val);
void _longjmp(void *env, int val);
}  // extern "C"
#endif  // __APPLE__

#if defined(_WIN32) && !defined(_DLL)
// FIXME: We might want to use these on Mac too.
extern "C" {
int memcmp(const void *b1, const void *b2, size_t sz);
void* memmove(void *d, const void *s, size_t sz);
void* memcpy(void *d, const void *s, size_t sz);
void* memset(void *b, int c, size_t sz);

char* strchr(const char *s, char c);
char* strcat(char *d, const char* s);  // NOLINT
char* strncat(char *d, const char* s, size_t sz);
char* strcpy(char *d, const char* s);  // NOLINT
char* strncpy(char *d, const char* s, size_t sz);
int strcmp(const char *s1, const char* s2);
int strncmp(const char *s1, const char* s2, size_t sz);
size_t strnlen(const char *s1, size_t sz);

void longjmp(void* env, int value);

__declspec(dllimport)
void* __stdcall CreateThread(void *sec, size_t st, void* start,
                             void *arg, DWORD fl, DWORD *id);
}  // extern "C"
#endif

namespace __asan {

// Instruments read/write access to a single byte in memory.
// On error calls __asan_report_error, which aborts the program.
#define ACCESS_ADDRESS(address, isWrite)   do {         \
  if (AddressIsPoisoned(address)) {                     \
    GET_CURRENT_PC_BP_SP;                               \
    __asan_report_error(pc, bp, sp, address, isWrite, /* access_size */ 1); \
  } \
} while (0)

// We implement ACCESS_MEMORY_RANGE, ASAN_READ_RANGE,
// and ASAN_WRITE_RANGE as macro instead of function so
// that no extra frames are created, and stack trace contains
// relevant information only.

// Instruments read/write access to a memory range.
// More complex implementation is possible, for now just
// checking the first and the last byte of a range.
#define ACCESS_MEMORY_RANGE(offset, size, isWrite) do { \
  if (size > 0) { \
    uintptr_t ptr = (uintptr_t)(offset); \
    ACCESS_ADDRESS(ptr, isWrite); \
    ACCESS_ADDRESS(ptr + (size) - 1, isWrite); \
  } \
} while (0)

#define ASAN_READ_RANGE(offset, size) do { \
  ACCESS_MEMORY_RANGE(offset, size, false); \
} while (0)

#define ASAN_WRITE_RANGE(offset, size) do { \
  ACCESS_MEMORY_RANGE(offset, size, true); \
} while (0)

// Behavior of functions like "memcpy" or "strcpy" is undefined
// if memory intervals overlap. We report error in this case.
// Macro is used to avoid creation of new frames.
static inline bool RangesOverlap(const char *offset1, size_t length1,
                                 const char *offset2, size_t length2) {
  return !((offset1 + length1 <= offset2) || (offset2 + length2 <= offset1));
}
#define CHECK_RANGES_OVERLAP(name, _offset1, length1, _offset2, length2) do { \
  const char *offset1 = (const char*)_offset1; \
  const char *offset2 = (const char*)_offset2; \
  if (RangesOverlap(offset1, length1, offset2, length2)) { \
    Report("ERROR: AddressSanitizer %s-param-overlap: " \
           "memory ranges [%p,%p) and [%p, %p) overlap\n", \
           name, offset1, offset1 + length1, offset2, offset2 + length2); \
    PRINT_CURRENT_STACK(); \
    ShowStatsAndAbort(); \
  } \
} while (0)

#define ENSURE_ASAN_INITED() do { \
  CHECK(!asan_init_is_running); \
  if (!asan_inited) { \
    __asan_init(); \
  } \
} while (0)

static inline bool IsSpace(int c) {
  return (c == ' ') || (c == '\n') || (c == '\t') ||
         (c == '\f') || (c == '\r') || (c == '\v');
}

static inline bool IsDigit(int c) {
  return (c >= '0') && (c <= '9');
}

static inline int ToLower(int c) {
  return (c >= 'A' && c <= 'Z') ? (c + 'a' - 'A') : c;
}

// ---------------------- Internal string functions ---------------- {{{1

int64_t internal_simple_strtoll(const char *nptr, char **endptr, int base) {
  CHECK(base == 10);
  while (IsSpace(*nptr)) nptr++;
  int sgn = 1;
  uint64_t res = 0;
  bool have_digits = false;
  char *old_nptr = (char*)nptr;
  if (*nptr == '+') {
    sgn = 1;
    nptr++;
  } else if (*nptr == '-') {
    sgn = -1;
    nptr++;
  }
  while (IsDigit(*nptr)) {
    res = (res <= UINT64_MAX / 10) ? res * 10 : UINT64_MAX;
    int digit = ((*nptr) - '0');
    res = (res <= UINT64_MAX - digit) ? res + digit : UINT64_MAX;
    have_digits = true;
    nptr++;
  }
  if (endptr != NULL) {
    *endptr = (have_digits) ? (char*)nptr : old_nptr;
  }
  if (sgn > 0) {
    return (int64_t)(Min((uint64_t)INT64_MAX, res));
  } else {
    return (res > INT64_MAX) ? INT64_MIN : ((int64_t)res * -1);
  }
}

int64_t internal_atoll(const char *nptr) {
  return internal_simple_strtoll(nptr, (char**)NULL, 10);
}

size_t internal_strlen(const char *s) {
  size_t i = 0;
  while (s[i]) i++;
  return i;
}

size_t internal_strnlen(const char *s, size_t maxlen) {
#ifndef __APPLE__
  if (REAL(strnlen) != NULL) {
    return REAL(strnlen)(s, maxlen);
  }
#endif
  size_t i = 0;
  while (i < maxlen && s[i]) i++;
  return i;
}

char* internal_strchr(const char *s, int c) {
  while (true) {
    if (*s == (char)c)
      return (char*)s;
    if (*s == 0)
      return NULL;
    s++;
  }
}

void* internal_memchr(const void* s, int c, size_t n) {
  const char* t = (char*)s;
  for (size_t i = 0; i < n; ++i, ++t)
    if (*t == c)
      return (void*)t;
  return NULL;
}

int internal_memcmp(const void* s1, const void* s2, size_t n) {
  const char* t1 = (char*)s1;
  const char* t2 = (char*)s2;
  for (size_t i = 0; i < n; ++i, ++t1, ++t2)
    if (*t1 != *t2)
      return *t1 < *t2 ? -1 : 1;
  return 0;
}

char *internal_strstr(const char *haystack, const char *needle) {
  // This is O(N^2), but we are not using it in hot places.
  size_t len1 = internal_strlen(haystack);
  size_t len2 = internal_strlen(needle);
  if (len1 < len2) return 0;
  for (size_t pos = 0; pos <= len1 - len2; pos++) {
    if (internal_memcmp(haystack + pos, needle, len2) == 0)
      return (char*)haystack + pos;
  }
  return 0;
}

char *internal_strncat(char *dst, const char *src, size_t n) {
  size_t len = internal_strlen(dst);
  size_t i;
  for (i = 0; i < n && src[i]; i++)
    dst[len + i] = src[i];
  dst[len + i] = 0;
  return dst;
}

int internal_strcmp(const char *s1, const char *s2) {
  while (true) {
    unsigned c1 = *s1;
    unsigned c2 = *s2;
    if (c1 != c2) return (c1 < c2) ? -1 : 1;
    if (c1 == 0) break;
    s1++;
    s2++;
  }
  return 0;
}

char *internal_strncpy(char *dst, const char *src, size_t n) {
  size_t i;
  for (i = 0; i < n && src[i]; i++)
    dst[i] = src[i];
  return dst;
}

}  // namespace __asan

// ---------------------- Wrappers ---------------- {{{1
using namespace __asan;  // NOLINT

#define OPERATOR_NEW_BODY \
  GET_STACK_TRACE_HERE_FOR_MALLOC;\
  return asan_memalign(0, size, &stack);

#ifdef ANDROID
void *operator new(size_t size) { OPERATOR_NEW_BODY; }
void *operator new[](size_t size) { OPERATOR_NEW_BODY; }
#else
void *operator new(size_t size) throw(std::bad_alloc) { OPERATOR_NEW_BODY; }
void *operator new[](size_t size) throw(std::bad_alloc) { OPERATOR_NEW_BODY; }
void *operator new(size_t size, std::nothrow_t const&) throw()
{ OPERATOR_NEW_BODY; }
void *operator new[](size_t size, std::nothrow_t const&) throw()
{ OPERATOR_NEW_BODY; }
#endif

#define OPERATOR_DELETE_BODY \
  GET_STACK_TRACE_HERE_FOR_FREE(ptr);\
  asan_free(ptr, &stack);

void operator delete(void *ptr) throw() { OPERATOR_DELETE_BODY; }
void operator delete[](void *ptr) throw() { OPERATOR_DELETE_BODY; }
void operator delete(void *ptr, std::nothrow_t const&) throw()
{ OPERATOR_DELETE_BODY; }
void operator delete[](void *ptr, std::nothrow_t const&) throw()
{ OPERATOR_DELETE_BODY;}

static thread_return_t THREAD_CALLING_CONV asan_thread_start(void *arg) {
  AsanThread *t = (AsanThread*)arg;
  asanThreadRegistry().SetCurrent(t);
  return t->ThreadStart();
}

#ifndef _WIN32
INTERCEPTOR(int, pthread_create, void *thread,
    void *attr, void *(*start_routine)(void*), void *arg) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  int current_tid = asanThreadRegistry().GetCurrentTidOrMinusOne();
  AsanThread *t = AsanThread::Create(current_tid, start_routine, arg, &stack);
  asanThreadRegistry().RegisterThread(t);
  return REAL(pthread_create)(thread, attr, asan_thread_start, t);
}
#endif  // !_WIN32

#if !defined(ANDROID) && !defined(_WIN32)
INTERCEPTOR(void*, signal, int signum, void *handler) {
  if (!AsanInterceptsSignal(signum)) {
    return REAL(signal)(signum, handler);
  }
  return NULL;
}

INTERCEPTOR(int, sigaction, int signum, const struct sigaction *act,
                            struct sigaction *oldact) {
  if (!AsanInterceptsSignal(signum)) {
    return REAL(sigaction)(signum, act, oldact);
  }
  return 0;
}
#endif  // !ANDROID && !_WIN32

INTERCEPTOR(void, longjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(longjmp)(env, val);
}

#if !defined(_WIN32)
INTERCEPTOR(void, _longjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(_longjmp)(env, val);
}

INTERCEPTOR(void, siglongjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(siglongjmp)(env, val);
}
#endif

#if ASAN_HAS_EXCEPTIONS == 1
#ifdef __APPLE__
extern "C" void __cxa_throw(void *a, void *b, void *c);
#endif  // __APPLE__

INTERCEPTOR(void, __cxa_throw, void *a, void *b, void *c) {
  CHECK(REAL(__cxa_throw));
  __asan_handle_no_return();
  REAL(__cxa_throw)(a, b, c);
}
#endif

// intercept mlock and friends.
// Since asan maps 16T of RAM, mlock is completely unfriendly to asan.
// All functions return 0 (success).
static void MlockIsUnsupported() {
  static bool printed = 0;
  if (printed) return;
  printed = true;
  Printf("INFO: AddressSanitizer ignores mlock/mlockall/munlock/munlockall\n");
}

extern "C" {
INTERCEPTOR_ATTRIBUTE
int mlock(const void *addr, size_t len) {
  MlockIsUnsupported();
  return 0;
}

INTERCEPTOR_ATTRIBUTE
int munlock(const void *addr, size_t len) {
  MlockIsUnsupported();
  return 0;
}

INTERCEPTOR_ATTRIBUTE
int mlockall(int flags) {
  MlockIsUnsupported();
  return 0;
}

INTERCEPTOR_ATTRIBUTE
int munlockall(void) {
  MlockIsUnsupported();
  return 0;
}
}  // extern "C"

static inline int CharCmp(unsigned char c1, unsigned char c2) {
  return (c1 == c2) ? 0 : (c1 < c2) ? -1 : 1;
}

static inline int CharCaseCmp(unsigned char c1, unsigned char c2) {
  int c1_low = ToLower(c1);
  int c2_low = ToLower(c2);
  return c1_low - c2_low;
}

INTERCEPTOR(int, memcmp, const void *a1, const void *a2, size_t size) {
  ENSURE_ASAN_INITED();
  unsigned char c1 = 0, c2 = 0;
  const unsigned char *s1 = (const unsigned char*)a1;
  const unsigned char *s2 = (const unsigned char*)a2;
  size_t i;
  for (i = 0; i < size; i++) {
    c1 = s1[i];
    c2 = s2[i];
    if (c1 != c2) break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, size));
  ASAN_READ_RANGE(s2, Min(i + 1, size));
  return CharCmp(c1, c2);
}

INTERCEPTOR(void*, memcpy, void *to, const void *from, size_t size) {
  // memcpy is called during __asan_init() from the internals
  // of printf(...).
  if (asan_init_is_running) {
    return REAL(memcpy)(to, from, size);
  }
  ENSURE_ASAN_INITED();
  if (FLAG_replace_intrin) {
    if (to != from) {
      // We do not treat memcpy with to==from as a bug.
      // See http://llvm.org/bugs/show_bug.cgi?id=11763.
      CHECK_RANGES_OVERLAP("memcpy", to, size, from, size);
    }
    ASAN_WRITE_RANGE(from, size);
    ASAN_READ_RANGE(to, size);
  }
  return REAL(memcpy)(to, from, size);
}

INTERCEPTOR(void*, memmove, void *to, const void *from, size_t size) {
  ENSURE_ASAN_INITED();
  if (FLAG_replace_intrin) {
    ASAN_WRITE_RANGE(from, size);
    ASAN_READ_RANGE(to, size);
  }
  return REAL(memmove)(to, from, size);
}

INTERCEPTOR(void*, memset, void *block, int c, size_t size) {
  // memset is called inside INTERCEPT_FUNCTION on Mac.
  if (asan_init_is_running) {
    return REAL(memset)(block, c, size);
  }
  ENSURE_ASAN_INITED();
  if (FLAG_replace_intrin) {
    ASAN_WRITE_RANGE(block, size);
  }
  return REAL(memset)(block, c, size);
}

INTERCEPTOR(char*, strchr, const char *str, int c) {
  ENSURE_ASAN_INITED();
  char *result = REAL(strchr)(str, c);
  if (FLAG_replace_str) {
    size_t bytes_read = (result ? result - str : REAL(strlen)(str)) + 1;
    ASAN_READ_RANGE(str, bytes_read);
  }
  return result;
}

#ifdef __linux__
INTERCEPTOR(char*, index, const char *string, int c)
  ALIAS(WRAPPER_NAME(strchr));
#else
DEFINE_REAL(char*, index, const char *string, int c);
#endif

#ifdef ANDROID
DEFINE_REAL(int, sigaction, int signum, const struct sigaction *act,
    struct sigaction *oldact);
#endif

INTERCEPTOR(int, strcasecmp, const char *s1, const char *s2) {
  ENSURE_ASAN_INITED();
  unsigned char c1, c2;
  size_t i;
  for (i = 0; ; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, i + 1);
  ASAN_READ_RANGE(s2, i + 1);
  return CharCaseCmp(c1, c2);
}

INTERCEPTOR(char*, strcat, char *to, const char *from) {  // NOLINT
  ENSURE_ASAN_INITED();
  if (FLAG_replace_str) {
    size_t from_length = REAL(strlen)(from);
    ASAN_READ_RANGE(from, from_length + 1);
    if (from_length > 0) {
      size_t to_length = REAL(strlen)(to);
      ASAN_READ_RANGE(to, to_length);
      ASAN_WRITE_RANGE(to + to_length, from_length + 1);
      CHECK_RANGES_OVERLAP("strcat", to, to_length + 1, from, from_length + 1);
    }
  }
  return REAL(strcat)(to, from);  // NOLINT
}

INTERCEPTOR(int, strcmp, const char *s1, const char *s2) {
  if (!asan_inited) {
    return internal_strcmp(s1, s2);
  }
  unsigned char c1, c2;
  size_t i;
  for (i = 0; ; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, i + 1);
  ASAN_READ_RANGE(s2, i + 1);
  return CharCmp(c1, c2);
}

INTERCEPTOR(char*, strcpy, char *to, const char *from) {  // NOLINT
  // strcpy is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return REAL(strcpy)(to, from);  // NOLINT
  }
  ENSURE_ASAN_INITED();
  if (FLAG_replace_str) {
    size_t from_size = REAL(strlen)(from) + 1;
    CHECK_RANGES_OVERLAP("strcpy", to, from_size, from, from_size);
    ASAN_READ_RANGE(from, from_size);
    ASAN_WRITE_RANGE(to, from_size);
  }
  return REAL(strcpy)(to, from);  // NOLINT
}

INTERCEPTOR(char*, strdup, const char *s) {
  ENSURE_ASAN_INITED();
  if (FLAG_replace_str) {
    size_t length = REAL(strlen)(s);
    ASAN_READ_RANGE(s, length + 1);
  }
  return REAL(strdup)(s);
}

INTERCEPTOR(size_t, strlen, const char *s) {
  // strlen is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return REAL(strlen)(s);
  }
  ENSURE_ASAN_INITED();
  size_t length = REAL(strlen)(s);
  if (FLAG_replace_str) {
    ASAN_READ_RANGE(s, length + 1);
  }
  return length;
}

INTERCEPTOR(int, strncasecmp, const char *s1, const char *s2, size_t n) {
  ENSURE_ASAN_INITED();
  unsigned char c1 = 0, c2 = 0;
  size_t i;
  for (i = 0; i < n; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, n));
  ASAN_READ_RANGE(s2, Min(i + 1, n));
  return CharCaseCmp(c1, c2);
}

INTERCEPTOR(int, strncmp, const char *s1, const char *s2, size_t size) {
  // strncmp is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return REAL(strncmp)(s1, s2, size);
  }
  unsigned char c1 = 0, c2 = 0;
  size_t i;
  for (i = 0; i < size; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, size));
  ASAN_READ_RANGE(s2, Min(i + 1, size));
  return CharCmp(c1, c2);
}

INTERCEPTOR(char*, strncpy, char *to, const char *from, size_t size) {
  ENSURE_ASAN_INITED();
  if (FLAG_replace_str) {
    size_t from_size = Min(size, internal_strnlen(from, size) + 1);
    CHECK_RANGES_OVERLAP("strncpy", to, from_size, from, from_size);
    ASAN_READ_RANGE(from, from_size);
    ASAN_WRITE_RANGE(to, size);
  }
  return REAL(strncpy)(to, from, size);
}

#ifndef __APPLE__
INTERCEPTOR(size_t, strnlen, const char *s, size_t maxlen) {
  ENSURE_ASAN_INITED();
  size_t length = REAL(strnlen)(s, maxlen);
  if (FLAG_replace_str) {
    ASAN_READ_RANGE(s, Min(length + 1, maxlen));
  }
  return length;
}
#endif

#if defined(_WIN32)
INTERCEPTOR_WINAPI(DWORD, CreateThread,
                   void* security, size_t stack_size,
                   DWORD (__stdcall *start_routine)(void*), void* arg,
                   DWORD flags, void* tid) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  int current_tid = asanThreadRegistry().GetCurrentTidOrMinusOne();
  AsanThread *t = AsanThread::Create(current_tid, start_routine, arg, &stack);
  asanThreadRegistry().RegisterThread(t);
  return REAL(CreateThread)(security, stack_size,
                            asan_thread_start, t, flags, tid);
}

namespace __asan {
void InitializeWindowsInterceptors() {
  CHECK(INTERCEPT_FUNCTION(CreateThread));
}

}  // namespace __asan
#endif

// ---------------------- InitializeAsanInterceptors ---------------- {{{1
namespace __asan {
void InitializeAsanInterceptors() {
  static bool was_called_once;
  CHECK(was_called_once == false);
  was_called_once = true;
  // Intercept mem* functions.
  CHECK(INTERCEPT_FUNCTION(memcmp));
  CHECK(INTERCEPT_FUNCTION(memmove));
  CHECK(INTERCEPT_FUNCTION(memset));
  if (PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE) {
    CHECK(INTERCEPT_FUNCTION(memcpy));
  } else {
    REAL(memcpy) = REAL(memmove);
  }

  // Intercept str* functions.
  CHECK(INTERCEPT_FUNCTION(strcat));  // NOLINT
  CHECK(INTERCEPT_FUNCTION(strchr));
  CHECK(INTERCEPT_FUNCTION(strcmp));
  CHECK(INTERCEPT_FUNCTION(strcpy));  // NOLINT
  CHECK(INTERCEPT_FUNCTION(strlen));
  CHECK(INTERCEPT_FUNCTION(strncmp));
  CHECK(INTERCEPT_FUNCTION(strncpy));
#if !defined(_WIN32)
  CHECK(INTERCEPT_FUNCTION(strcasecmp));
  CHECK(INTERCEPT_FUNCTION(strdup));
  CHECK(INTERCEPT_FUNCTION(strncasecmp));
# ifndef __APPLE__
  CHECK(INTERCEPT_FUNCTION(index));
# else
  CHECK(OVERRIDE_FUNCTION(index, WRAP(strchr)));
# endif
#endif
#if !defined(__APPLE__)
  CHECK(INTERCEPT_FUNCTION(strnlen));
#endif

  // Intecept signal- and jump-related functions.
  CHECK(INTERCEPT_FUNCTION(longjmp));
#if !defined(ANDROID) && !defined(_WIN32)
  CHECK(INTERCEPT_FUNCTION(sigaction));
  CHECK(INTERCEPT_FUNCTION(signal));
#endif

#if !defined(_WIN32)
  CHECK(INTERCEPT_FUNCTION(_longjmp));
  INTERCEPT_FUNCTION(__cxa_throw);
# if !defined(__APPLE__)
  // On Darwin siglongjmp tailcalls longjmp, so we don't want to intercept it
  // there.
  CHECK(INTERCEPT_FUNCTION(siglongjmp));
# endif
#endif

  // Intercept threading-related functions
#if !defined(_WIN32)
  CHECK(INTERCEPT_FUNCTION(pthread_create));
# if defined(__APPLE__)
  // We don't need to intercept pthread_workqueue_additem_np() to support the
  // libdispatch API, but it helps us to debug the unsupported functions. Let's
  // intercept it only during verbose runs.
  if (FLAG_v >= 2) {
    CHECK(INTERCEPT_FUNCTION(pthread_workqueue_additem_np));
  }
# endif
#endif

  // Some Windows-specific interceptors.
#if defined(_WIN32)
  InitializeWindowsInterceptors();
#endif

  // Some Mac-specific interceptors.
#if defined(__APPLE__)
  CHECK(INTERCEPT_FUNCTION(dispatch_async_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_sync_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_after_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_barrier_async_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_group_async_f));

  // Normally CFStringCreateCopy should not copy constant CF strings.
  // Replacing the default CFAllocator causes constant strings to be copied
  // rather than just returned, which leads to bugs in big applications like
  // Chromium and WebKit, see
  // http://code.google.com/p/address-sanitizer/issues/detail?id=10
  // Until this problem is fixed we need to check that the string is
  // non-constant before calling CFStringCreateCopy.
  CHECK(INTERCEPT_FUNCTION(CFStringCreateCopy));
#endif

  if (FLAG_v > 0) {
    Printf("AddressSanitizer: libc interceptors initialized\n");
  }
}

}  // namespace __asan
