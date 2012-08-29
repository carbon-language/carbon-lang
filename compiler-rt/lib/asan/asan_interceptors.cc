//===-- asan_interceptors.cc ----------------------------------------------===//
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
#include "asan_intercepted_functions.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread_registry.h"
#include "interception/interception.h"
#include "sanitizer/asan_interface.h"
#include "sanitizer_common/sanitizer_libc.h"

namespace __asan {

// Instruments read/write access to a single byte in memory.
// On error calls __asan_report_error, which aborts the program.
#define ACCESS_ADDRESS(address, isWrite)   do {         \
  if (!AddrIsInMem(address) || AddressIsPoisoned(address)) {                \
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
    uptr ptr = (uptr)(offset); \
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
static inline bool RangesOverlap(const char *offset1, uptr length1,
                                 const char *offset2, uptr length2) {
  return !((offset1 + length1 <= offset2) || (offset2 + length2 <= offset1));
}
#define CHECK_RANGES_OVERLAP(name, _offset1, length1, _offset2, length2) do { \
  const char *offset1 = (const char*)_offset1; \
  const char *offset2 = (const char*)_offset2; \
  if (RangesOverlap(offset1, length1, offset2, length2)) { \
    GET_STACK_TRACE_HERE(kStackTraceMax); \
    ReportStringFunctionMemoryRangesOverlap(name, offset1, length1, \
                                            offset2, length2, &stack); \
  } \
} while (0)

#define ENSURE_ASAN_INITED() do { \
  CHECK(!asan_init_is_running); \
  if (!asan_inited) { \
    __asan_init(); \
  } \
} while (0)

static inline uptr MaybeRealStrnlen(const char *s, uptr maxlen) {
#if ASAN_INTERCEPT_STRNLEN
  if (REAL(strnlen) != 0) {
    return REAL(strnlen)(s, maxlen);
  }
#endif
  return internal_strnlen(s, maxlen);
}

}  // namespace __asan

// ---------------------- Wrappers ---------------- {{{1
using namespace __asan;  // NOLINT

static thread_return_t THREAD_CALLING_CONV asan_thread_start(void *arg) {
  AsanThread *t = (AsanThread*)arg;
  asanThreadRegistry().SetCurrent(t);
  return t->ThreadStart();
}

#if ASAN_INTERCEPT_PTHREAD_CREATE
INTERCEPTOR(int, pthread_create, void *thread,
    void *attr, void *(*start_routine)(void*), void *arg) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  u32 current_tid = asanThreadRegistry().GetCurrentTidOrInvalid();
  AsanThread *t = AsanThread::Create(current_tid, start_routine, arg, &stack);
  asanThreadRegistry().RegisterThread(t);
  return REAL(pthread_create)(thread, attr, asan_thread_start, t);
}
#endif  // ASAN_INTERCEPT_PTHREAD_CREATE

#if ASAN_INTERCEPT_SIGNAL_AND_SIGACTION
INTERCEPTOR(void*, signal, int signum, void *handler) {
  if (!AsanInterceptsSignal(signum)) {
    return REAL(signal)(signum, handler);
  }
  return 0;
}

INTERCEPTOR(int, sigaction, int signum, const struct sigaction *act,
                            struct sigaction *oldact) {
  if (!AsanInterceptsSignal(signum)) {
    return REAL(sigaction)(signum, act, oldact);
  }
  return 0;
}
#elif ASAN_POSIX
// We need to have defined REAL(sigaction) on posix systems.
DEFINE_REAL(int, sigaction, int signum, const struct sigaction *act,
    struct sigaction *oldact);
#endif  // ASAN_INTERCEPT_SIGNAL_AND_SIGACTION

INTERCEPTOR(void, longjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(longjmp)(env, val);
}

#if ASAN_INTERCEPT__LONGJMP
INTERCEPTOR(void, _longjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(_longjmp)(env, val);
}
#endif

#if ASAN_INTERCEPT_SIGLONGJMP
INTERCEPTOR(void, siglongjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(siglongjmp)(env, val);
}
#endif

#if ASAN_INTERCEPT___CXA_THROW
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
}  // extern "C"

static inline int CharCmp(unsigned char c1, unsigned char c2) {
  return (c1 == c2) ? 0 : (c1 < c2) ? -1 : 1;
}

static inline int CharCaseCmp(unsigned char c1, unsigned char c2) {
  int c1_low = ToLower(c1);
  int c2_low = ToLower(c2);
  return c1_low - c2_low;
}

INTERCEPTOR(int, memcmp, const void *a1, const void *a2, uptr size) {
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(memcmp)(a1, a2, size);
#endif
  ENSURE_ASAN_INITED();
  unsigned char c1 = 0, c2 = 0;
  const unsigned char *s1 = (const unsigned char*)a1;
  const unsigned char *s2 = (const unsigned char*)a2;
  uptr i;
  for (i = 0; i < size; i++) {
    c1 = s1[i];
    c2 = s2[i];
    if (c1 != c2) break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, size));
  ASAN_READ_RANGE(s2, Min(i + 1, size));
  return CharCmp(c1, c2);
}

INTERCEPTOR(void*, memcpy, void *to, const void *from, uptr size) {
  // memcpy is called during __asan_init() from the internals
  // of printf(...).
  if (asan_init_is_running) {
    return REAL(memcpy)(to, from, size);
  }
  ENSURE_ASAN_INITED();
  if (flags()->replace_intrin) {
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

INTERCEPTOR(void*, memmove, void *to, const void *from, uptr size) {
  if (asan_init_is_running) {
    return REAL(memmove)(to, from, size);
  }
  ENSURE_ASAN_INITED();
  if (flags()->replace_intrin) {
    ASAN_WRITE_RANGE(from, size);
    ASAN_READ_RANGE(to, size);
  }
  return REAL(memmove)(to, from, size);
}

INTERCEPTOR(void*, memset, void *block, int c, uptr size) {
  // memset is called inside Printf.
  if (asan_init_is_running) {
    return REAL(memset)(block, c, size);
  }
  ENSURE_ASAN_INITED();
  if (flags()->replace_intrin) {
    ASAN_WRITE_RANGE(block, size);
  }
  return REAL(memset)(block, c, size);
}

INTERCEPTOR(char*, strchr, const char *str, int c) {
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(strchr)(str, c);
#endif
  ENSURE_ASAN_INITED();
  char *result = REAL(strchr)(str, c);
  if (flags()->replace_str) {
    uptr bytes_read = (result ? result - str : REAL(strlen)(str)) + 1;
    ASAN_READ_RANGE(str, bytes_read);
  }
  return result;
}

#if ASAN_INTERCEPT_INDEX
# if ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX
INTERCEPTOR(char*, index, const char *string, int c)
  ALIAS(WRAPPER_NAME(strchr));
# else
DEFINE_REAL(char*, index, const char *string, int c)
# endif
#endif  // ASAN_INTERCEPT_INDEX

// For both strcat() and strncat() we need to check the validity of |to|
// argument irrespective of the |from| length.
INTERCEPTOR(char*, strcat, char *to, const char *from) {  // NOLINT
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_length = REAL(strlen)(from);
    ASAN_READ_RANGE(from, from_length + 1);
    uptr to_length = REAL(strlen)(to);
    ASAN_READ_RANGE(to, to_length);
    ASAN_WRITE_RANGE(to + to_length, from_length + 1);
    // If the copying actually happens, the |from| string should not overlap
    // with the resulting string starting at |to|, which has a length of
    // to_length + from_length + 1.
    if (from_length > 0) {
      CHECK_RANGES_OVERLAP("strcat", to, from_length + to_length + 1,
                           from, from_length + 1);
    }
  }
  return REAL(strcat)(to, from);  // NOLINT
}

INTERCEPTOR(char*, strncat, char *to, const char *from, uptr size) {
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_length = MaybeRealStrnlen(from, size);
    uptr copy_length = Min(size, from_length + 1);
    ASAN_READ_RANGE(from, copy_length);
    uptr to_length = REAL(strlen)(to);
    ASAN_READ_RANGE(to, to_length);
    ASAN_WRITE_RANGE(to + to_length, from_length + 1);
    if (from_length > 0) {
      CHECK_RANGES_OVERLAP("strncat", to, to_length + copy_length + 1,
                           from, copy_length);
    }
  }
  return REAL(strncat)(to, from, size);
}

INTERCEPTOR(int, strcmp, const char *s1, const char *s2) {
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(strcmp)(s1, s2);
#endif
  if (asan_init_is_running) {
    return REAL(strcmp)(s1, s2);
  }
  ENSURE_ASAN_INITED();
  unsigned char c1, c2;
  uptr i;
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
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(strcpy)(to, from);  // NOLINT
#endif
  // strcpy is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return REAL(strcpy)(to, from);  // NOLINT
  }
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_size = REAL(strlen)(from) + 1;
    CHECK_RANGES_OVERLAP("strcpy", to, from_size, from, from_size);
    ASAN_READ_RANGE(from, from_size);
    ASAN_WRITE_RANGE(to, from_size);
  }
  return REAL(strcpy)(to, from);  // NOLINT
}

#if ASAN_INTERCEPT_STRDUP
INTERCEPTOR(char*, strdup, const char *s) {
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(strdup)(s);
#endif
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr length = REAL(strlen)(s);
    ASAN_READ_RANGE(s, length + 1);
  }
  return REAL(strdup)(s);
}
#endif

INTERCEPTOR(uptr, strlen, const char *s) {
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(strlen)(s);
#endif
  // strlen is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return REAL(strlen)(s);
  }
  ENSURE_ASAN_INITED();
  uptr length = REAL(strlen)(s);
  if (flags()->replace_str) {
    ASAN_READ_RANGE(s, length + 1);
  }
  return length;
}

#if ASAN_INTERCEPT_STRCASECMP_AND_STRNCASECMP
INTERCEPTOR(int, strcasecmp, const char *s1, const char *s2) {
  ENSURE_ASAN_INITED();
  unsigned char c1, c2;
  uptr i;
  for (i = 0; ; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, i + 1);
  ASAN_READ_RANGE(s2, i + 1);
  return CharCaseCmp(c1, c2);
}

INTERCEPTOR(int, strncasecmp, const char *s1, const char *s2, uptr n) {
  ENSURE_ASAN_INITED();
  unsigned char c1 = 0, c2 = 0;
  uptr i;
  for (i = 0; i < n; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (CharCaseCmp(c1, c2) != 0 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, n));
  ASAN_READ_RANGE(s2, Min(i + 1, n));
  return CharCaseCmp(c1, c2);
}
#endif  // ASAN_INTERCEPT_STRCASECMP_AND_STRNCASECMP

INTERCEPTOR(int, strncmp, const char *s1, const char *s2, uptr size) {
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(strncmp)(s1, s2, size);
#endif
  // strncmp is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return REAL(strncmp)(s1, s2, size);
  }
  ENSURE_ASAN_INITED();
  unsigned char c1 = 0, c2 = 0;
  uptr i;
  for (i = 0; i < size; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  ASAN_READ_RANGE(s1, Min(i + 1, size));
  ASAN_READ_RANGE(s2, Min(i + 1, size));
  return CharCmp(c1, c2);
}

INTERCEPTOR(char*, strncpy, char *to, const char *from, uptr size) {
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_size = Min(size, MaybeRealStrnlen(from, size) + 1);
    CHECK_RANGES_OVERLAP("strncpy", to, from_size, from, from_size);
    ASAN_READ_RANGE(from, from_size);
    ASAN_WRITE_RANGE(to, size);
  }
  return REAL(strncpy)(to, from, size);
}

#if ASAN_INTERCEPT_STRNLEN
INTERCEPTOR(uptr, strnlen, const char *s, uptr maxlen) {
  ENSURE_ASAN_INITED();
  uptr length = REAL(strnlen)(s, maxlen);
  if (flags()->replace_str) {
    ASAN_READ_RANGE(s, Min(length + 1, maxlen));
  }
  return length;
}
#endif  // ASAN_INTERCEPT_STRNLEN

static inline bool IsValidStrtolBase(int base) {
  return (base == 0) || (2 <= base && base <= 36);
}

static inline void FixRealStrtolEndptr(const char *nptr, char **endptr) {
  CHECK(endptr != 0);
  if (nptr == *endptr) {
    // No digits were found at strtol call, we need to find out the last
    // symbol accessed by strtoll on our own.
    // We get this symbol by skipping leading blanks and optional +/- sign.
    while (IsSpace(*nptr)) nptr++;
    if (*nptr == '+' || *nptr == '-') nptr++;
    *endptr = (char*)nptr;
  }
  CHECK(*endptr >= nptr);
}

INTERCEPTOR(long, strtol, const char *nptr,  // NOLINT
            char **endptr, int base) {
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(strtol)(nptr, endptr, base);
  }
  char *real_endptr;
  long result = REAL(strtol)(nptr, &real_endptr, base);  // NOLINT
  if (endptr != 0) {
    *endptr = real_endptr;
  }
  if (IsValidStrtolBase(base)) {
    FixRealStrtolEndptr(nptr, &real_endptr);
    ASAN_READ_RANGE(nptr, (real_endptr - nptr) + 1);
  }
  return result;
}

INTERCEPTOR(int, atoi, const char *nptr) {
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(atoi)(nptr);
#endif
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(atoi)(nptr);
  }
  char *real_endptr;
  // "man atoi" tells that behavior of atoi(nptr) is the same as
  // strtol(nptr, 0, 10), i.e. it sets errno to ERANGE if the
  // parsed integer can't be stored in *long* type (even if it's
  // different from int). So, we just imitate this behavior.
  int result = REAL(strtol)(nptr, &real_endptr, 10);
  FixRealStrtolEndptr(nptr, &real_endptr);
  ASAN_READ_RANGE(nptr, (real_endptr - nptr) + 1);
  return result;
}

INTERCEPTOR(long, atol, const char *nptr) {  // NOLINT
#if MAC_INTERPOSE_FUNCTIONS
  if (!asan_inited) return REAL(atol)(nptr);
#endif
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(atol)(nptr);
  }
  char *real_endptr;
  long result = REAL(strtol)(nptr, &real_endptr, 10);  // NOLINT
  FixRealStrtolEndptr(nptr, &real_endptr);
  ASAN_READ_RANGE(nptr, (real_endptr - nptr) + 1);
  return result;
}

#if ASAN_INTERCEPT_ATOLL_AND_STRTOLL
INTERCEPTOR(long long, strtoll, const char *nptr,  // NOLINT
            char **endptr, int base) {
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(strtoll)(nptr, endptr, base);
  }
  char *real_endptr;
  long long result = REAL(strtoll)(nptr, &real_endptr, base);  // NOLINT
  if (endptr != 0) {
    *endptr = real_endptr;
  }
  // If base has unsupported value, strtoll can exit with EINVAL
  // without reading any characters. So do additional checks only
  // if base is valid.
  if (IsValidStrtolBase(base)) {
    FixRealStrtolEndptr(nptr, &real_endptr);
    ASAN_READ_RANGE(nptr, (real_endptr - nptr) + 1);
  }
  return result;
}

INTERCEPTOR(long long, atoll, const char *nptr) {  // NOLINT
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(atoll)(nptr);
  }
  char *real_endptr;
  long long result = REAL(strtoll)(nptr, &real_endptr, 10);  // NOLINT
  FixRealStrtolEndptr(nptr, &real_endptr);
  ASAN_READ_RANGE(nptr, (real_endptr - nptr) + 1);
  return result;
}
#endif  // ASAN_INTERCEPT_ATOLL_AND_STRTOLL

#define ASAN_INTERCEPT_FUNC(name) do { \
      if (!INTERCEPT_FUNCTION(name) && flags()->verbosity > 0) \
        Report("AddressSanitizer: failed to intercept '" #name "'\n"); \
    } while (0)

#if defined(_WIN32)
INTERCEPTOR_WINAPI(DWORD, CreateThread,
                   void* security, uptr stack_size,
                   DWORD (__stdcall *start_routine)(void*), void* arg,
                   DWORD flags, void* tid) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  u32 current_tid = asanThreadRegistry().GetCurrentTidOrInvalid();
  AsanThread *t = AsanThread::Create(current_tid, start_routine, arg, &stack);
  asanThreadRegistry().RegisterThread(t);
  return REAL(CreateThread)(security, stack_size,
                            asan_thread_start, t, flags, tid);
}

namespace __asan {
void InitializeWindowsInterceptors() {
  ASAN_INTERCEPT_FUNC(CreateThread);
}

}  // namespace __asan
#endif

// ---------------------- InitializeAsanInterceptors ---------------- {{{1
namespace __asan {
void InitializeAsanInterceptors() {
  static bool was_called_once;
  CHECK(was_called_once == false);
  was_called_once = true;
#if MAC_INTERPOSE_FUNCTIONS
  return;
#endif
  // Intercept mem* functions.
  ASAN_INTERCEPT_FUNC(memcmp);
  ASAN_INTERCEPT_FUNC(memmove);
  ASAN_INTERCEPT_FUNC(memset);
  if (PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE) {
    ASAN_INTERCEPT_FUNC(memcpy);
  } else {
#if !MAC_INTERPOSE_FUNCTIONS
    // If we're using dynamic interceptors on Mac, these two are just plain
    // functions.
    *(uptr*)&REAL(memcpy) = (uptr)REAL(memmove);
#endif
  }

  // Intercept str* functions.
  ASAN_INTERCEPT_FUNC(strcat);  // NOLINT
  ASAN_INTERCEPT_FUNC(strchr);
  ASAN_INTERCEPT_FUNC(strcmp);
  ASAN_INTERCEPT_FUNC(strcpy);  // NOLINT
  ASAN_INTERCEPT_FUNC(strlen);
  ASAN_INTERCEPT_FUNC(strncat);
  ASAN_INTERCEPT_FUNC(strncmp);
  ASAN_INTERCEPT_FUNC(strncpy);
#if ASAN_INTERCEPT_STRCASECMP_AND_STRNCASECMP
  ASAN_INTERCEPT_FUNC(strcasecmp);
  ASAN_INTERCEPT_FUNC(strncasecmp);
#endif
#if ASAN_INTERCEPT_STRDUP
  ASAN_INTERCEPT_FUNC(strdup);
#endif
#if ASAN_INTERCEPT_STRNLEN
  ASAN_INTERCEPT_FUNC(strnlen);
#endif
#if ASAN_INTERCEPT_INDEX
# if ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX
  ASAN_INTERCEPT_FUNC(index);
# else
  CHECK(OVERRIDE_FUNCTION(index, WRAP(strchr)));
# endif
#endif

  ASAN_INTERCEPT_FUNC(atoi);
  ASAN_INTERCEPT_FUNC(atol);
  ASAN_INTERCEPT_FUNC(strtol);
#if ASAN_INTERCEPT_ATOLL_AND_STRTOLL
  ASAN_INTERCEPT_FUNC(atoll);
  ASAN_INTERCEPT_FUNC(strtoll);
#endif

#if ASAN_INTERCEPT_MLOCKX
  // Intercept mlock/munlock.
  ASAN_INTERCEPT_FUNC(mlock);
  ASAN_INTERCEPT_FUNC(munlock);
  ASAN_INTERCEPT_FUNC(mlockall);
  ASAN_INTERCEPT_FUNC(munlockall);
#endif

  // Intecept signal- and jump-related functions.
  ASAN_INTERCEPT_FUNC(longjmp);
#if ASAN_INTERCEPT_SIGNAL_AND_SIGACTION
  ASAN_INTERCEPT_FUNC(sigaction);
  ASAN_INTERCEPT_FUNC(signal);
#endif
#if ASAN_INTERCEPT__LONGJMP
  ASAN_INTERCEPT_FUNC(_longjmp);
#endif
#if ASAN_INTERCEPT_SIGLONGJMP
  ASAN_INTERCEPT_FUNC(siglongjmp);
#endif

  // Intercept exception handling functions.
#if ASAN_INTERCEPT___CXA_THROW
  INTERCEPT_FUNCTION(__cxa_throw);
#endif

  // Intercept threading-related functions
#if ASAN_INTERCEPT_PTHREAD_CREATE
  ASAN_INTERCEPT_FUNC(pthread_create);
#endif

  // Some Windows-specific interceptors.
#if defined(_WIN32)
  InitializeWindowsInterceptors();
#endif

  // Some Mac-specific interceptors.
#if defined(__APPLE__)
  InitializeMacInterceptors();
#endif

  if (flags()->verbosity > 0) {
    Report("AddressSanitizer: libc interceptors initialized\n");
  }
}

}  // namespace __asan
