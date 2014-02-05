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
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_poisoning.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "sanitizer_common/sanitizer_libc.h"

namespace __asan {

// Return true if we can quickly decide that the region is unpoisoned.
static inline bool QuickCheckForUnpoisonedRegion(uptr beg, uptr size) {
  if (size == 0) return true;
  if (size <= 32)
    return !AddressIsPoisoned(beg) &&
           !AddressIsPoisoned(beg + size - 1) &&
           !AddressIsPoisoned(beg + size / 2);
  return false;
}

// We implement ACCESS_MEMORY_RANGE, ASAN_READ_RANGE,
// and ASAN_WRITE_RANGE as macro instead of function so
// that no extra frames are created, and stack trace contains
// relevant information only.
// We check all shadow bytes.
#define ACCESS_MEMORY_RANGE(offset, size, isWrite) do {                 \
    uptr __offset = (uptr)(offset);                                     \
    uptr __size = (uptr)(size);                                         \
    uptr __bad = 0;                                                     \
    if (!QuickCheckForUnpoisonedRegion(__offset, __size) &&             \
        (__bad = __asan_region_is_poisoned(__offset, __size))) {        \
      GET_CURRENT_PC_BP_SP;                                             \
      __asan_report_error(pc, bp, sp, __bad, isWrite, __size);          \
    }                                                                   \
  } while (0)

#define ASAN_READ_RANGE(offset, size) ACCESS_MEMORY_RANGE(offset, size, false)
#define ASAN_WRITE_RANGE(offset, size) ACCESS_MEMORY_RANGE(offset, size, true)

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
    GET_STACK_TRACE_FATAL_HERE; \
    ReportStringFunctionMemoryRangesOverlap(name, offset1, length1, \
                                            offset2, length2, &stack); \
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

void SetThreadName(const char *name) {
  AsanThread *t = GetCurrentThread();
  if (t)
    asanThreadRegistry().SetThreadName(t->tid(), name);
}

int OnExit() {
  // FIXME: ask frontend whether we need to return failure.
  return 0;
}

}  // namespace __asan

// ---------------------- Wrappers ---------------- {{{1
using namespace __asan;  // NOLINT

DECLARE_REAL_AND_INTERCEPTOR(void *, malloc, uptr)
DECLARE_REAL_AND_INTERCEPTOR(void, free, void *)

#if !SANITIZER_MAC
#define ASAN_INTERCEPT_FUNC(name)                                        \
  do {                                                                   \
    if ((!INTERCEPT_FUNCTION(name) || !REAL(name)))                      \
      VReport(1, "AddressSanitizer: failed to intercept '" #name "'\n"); \
  } while (0)
#else
// OS X interceptors don't need to be initialized with INTERCEPT_FUNCTION.
#define ASAN_INTERCEPT_FUNC(name)
#endif  // SANITIZER_MAC

#define COMMON_INTERCEPT_FUNCTION(name) ASAN_INTERCEPT_FUNC(name)
#define COMMON_INTERCEPTOR_UNPOISON_PARAM(ctx, count) \
  do {                                                \
  } while (false)
#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  ASAN_WRITE_RANGE(ptr, size)
#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size) ASAN_READ_RANGE(ptr, size)
#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)                       \
  do {                                                                 \
    if (asan_init_is_running) return REAL(func)(__VA_ARGS__);          \
    ctx = 0;                                                           \
    (void) ctx;                                                        \
    if (SANITIZER_MAC && !asan_inited) return REAL(func)(__VA_ARGS__); \
    ENSURE_ASAN_INITED();                                              \
  } while (false)
#define COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd) \
  do {                                         \
  } while (false)
#define COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd) \
  do {                                         \
  } while (false)
#define COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, newfd) \
  do {                                                      \
  } while (false)
#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) SetThreadName(name)
// Should be asanThreadRegistry().SetThreadNameByUserId(thread, name)
// But asan does not remember UserId's for threads (pthread_t);
// and remembers all ever existed threads, so the linear search by UserId
// can be slow.
#define COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name) \
  do {                                                         \
  } while (false)
#define COMMON_INTERCEPTOR_BLOCK_REAL(name) REAL(name)
#define COMMON_INTERCEPTOR_ON_EXIT(ctx) OnExit()
#include "sanitizer_common/sanitizer_common_interceptors.inc"

#define COMMON_SYSCALL_PRE_READ_RANGE(p, s) ASAN_READ_RANGE(p, s)
#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s) ASAN_WRITE_RANGE(p, s)
#define COMMON_SYSCALL_POST_READ_RANGE(p, s) \
  do {                                       \
    (void)(p);                               \
    (void)(s);                               \
  } while (false)
#define COMMON_SYSCALL_POST_WRITE_RANGE(p, s) \
  do {                                        \
    (void)(p);                                \
    (void)(s);                                \
  } while (false)
#include "sanitizer_common/sanitizer_common_syscalls.inc"

static thread_return_t THREAD_CALLING_CONV asan_thread_start(void *arg) {
  AsanThread *t = (AsanThread*)arg;
  SetCurrentThread(t);
  return t->ThreadStart(GetTid());
}

#if ASAN_INTERCEPT_PTHREAD_CREATE
INTERCEPTOR(int, pthread_create, void *thread,
    void *attr, void *(*start_routine)(void*), void *arg) {
  EnsureMainThreadIDIsCorrect();
  // Strict init-order checking in thread-hostile.
  if (flags()->strict_init_order)
    StopInitOrderChecking();
  GET_STACK_TRACE_THREAD;
  int detached = 0;
  if (attr != 0)
    REAL(pthread_attr_getdetachstate)(attr, &detached);

  u32 current_tid = GetCurrentTidOrInvalid();
  AsanThread *t = AsanThread::Create(start_routine, arg);
  CreateThreadContextArgs args = { t, &stack };
  asanThreadRegistry().CreateThread(*(uptr*)t, detached, current_tid, &args);
  return REAL(pthread_create)(thread, attr, asan_thread_start, t);
}
#endif  // ASAN_INTERCEPT_PTHREAD_CREATE

#if ASAN_INTERCEPT_SIGNAL_AND_SIGACTION
INTERCEPTOR(void*, signal, int signum, void *handler) {
  if (!AsanInterceptsSignal(signum) ||
      common_flags()->allow_user_segv_handler) {
    return REAL(signal)(signum, handler);
  }
  return 0;
}

INTERCEPTOR(int, sigaction, int signum, const struct sigaction *act,
                            struct sigaction *oldact) {
  if (!AsanInterceptsSignal(signum) ||
      common_flags()->allow_user_segv_handler) {
    return REAL(sigaction)(signum, act, oldact);
  }
  return 0;
}

namespace __sanitizer {
int real_sigaction(int signum, const void *act, void *oldact) {
  return REAL(sigaction)(signum,
                         (struct sigaction *)act, (struct sigaction *)oldact);
}
}  // namespace __sanitizer

#elif SANITIZER_POSIX
// We need to have defined REAL(sigaction) on posix systems.
DEFINE_REAL(int, sigaction, int signum, const struct sigaction *act,
    struct sigaction *oldact)
#endif  // ASAN_INTERCEPT_SIGNAL_AND_SIGACTION

#if ASAN_INTERCEPT_SWAPCONTEXT
static void ClearShadowMemoryForContextStack(uptr stack, uptr ssize) {
  // Align to page size.
  uptr PageSize = GetPageSizeCached();
  uptr bottom = stack & ~(PageSize - 1);
  ssize += stack - bottom;
  ssize = RoundUpTo(ssize, PageSize);
  static const uptr kMaxSaneContextStackSize = 1 << 22;  // 4 Mb
  if (ssize && ssize <= kMaxSaneContextStackSize) {
    PoisonShadow(bottom, ssize, 0);
  }
}

INTERCEPTOR(int, swapcontext, struct ucontext_t *oucp,
            struct ucontext_t *ucp) {
  static bool reported_warning = false;
  if (!reported_warning) {
    Report("WARNING: ASan doesn't fully support makecontext/swapcontext "
           "functions and may produce false positives in some cases!\n");
    reported_warning = true;
  }
  // Clear shadow memory for new context (it may share stack
  // with current context).
  uptr stack, ssize;
  ReadContextStack(ucp, &stack, &ssize);
  ClearShadowMemoryForContextStack(stack, ssize);
  int res = REAL(swapcontext)(oucp, ucp);
  // swapcontext technically does not return, but program may swap context to
  // "oucp" later, that would look as if swapcontext() returned 0.
  // We need to clear shadow for ucp once again, as it may be in arbitrary
  // state.
  ClearShadowMemoryForContextStack(stack, ssize);
  return res;
}
#endif  // ASAN_INTERCEPT_SWAPCONTEXT

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
  static bool printed = false;
  if (printed) return;
  printed = true;
  VPrintf(1,
          "INFO: AddressSanitizer ignores "
          "mlock/mlockall/munlock/munlockall\n");
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

static inline int CharCmp(unsigned char c1, unsigned char c2) {
  return (c1 == c2) ? 0 : (c1 < c2) ? -1 : 1;
}

INTERCEPTOR(int, memcmp, const void *a1, const void *a2, uptr size) {
  if (!asan_inited) return internal_memcmp(a1, a2, size);
  ENSURE_ASAN_INITED();
  if (flags()->replace_intrin) {
    if (flags()->strict_memcmp) {
      // Check the entire regions even if the first bytes of the buffers are
      // different.
      ASAN_READ_RANGE(a1, size);
      ASAN_READ_RANGE(a2, size);
      // Fallthrough to REAL(memcmp) below.
    } else {
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
  }
  return REAL(memcmp(a1, a2, size));
}

#define MEMMOVE_BODY { \
  if (!asan_inited) return internal_memmove(to, from, size); \
  if (asan_init_is_running) { \
    return REAL(memmove)(to, from, size); \
  } \
  ENSURE_ASAN_INITED(); \
  if (flags()->replace_intrin) { \
    ASAN_READ_RANGE(from, size); \
    ASAN_WRITE_RANGE(to, size); \
  } \
  return internal_memmove(to, from, size); \
}

INTERCEPTOR(void*, memmove, void *to, const void *from, uptr size) MEMMOVE_BODY

INTERCEPTOR(void*, memcpy, void *to, const void *from, uptr size) {
#if !SANITIZER_MAC
  if (!asan_inited) return internal_memcpy(to, from, size);
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
    ASAN_READ_RANGE(from, size);
    ASAN_WRITE_RANGE(to, size);
  }
  // Interposing of resolver functions is broken on Mac OS 10.7 and 10.8, so
  // calling REAL(memcpy) here leads to infinite recursion.
  // See also http://code.google.com/p/address-sanitizer/issues/detail?id=116.
  return internal_memcpy(to, from, size);
#else
  // At least on 10.7 and 10.8 both memcpy() and memmove() are being replaced
  // with WRAP(memcpy). As a result, false positives are reported for memmove()
  // calls. If we just disable error reporting with
  // ASAN_OPTIONS=replace_intrin=0, memmove() is still replaced with
  // internal_memcpy(), which may lead to crashes, see
  // http://llvm.org/bugs/show_bug.cgi?id=16362.
  MEMMOVE_BODY
#endif  // !SANITIZER_MAC
}

INTERCEPTOR(void*, memset, void *block, int c, uptr size) {
  if (!asan_inited) return internal_memset(block, c, size);
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
  if (!asan_inited) return internal_strchr(str, c);
  // strchr is called inside create_purgeable_zone() when MallocGuardEdges=1 is
  // used.
  if (asan_init_is_running) {
    return REAL(strchr)(str, c);
  }
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
#  if SANITIZER_MAC
DECLARE_REAL(char*, index, const char *string, int c)
OVERRIDE_FUNCTION(index, strchr);
#  else
DEFINE_REAL(char*, index, const char *string, int c)
#  endif
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

INTERCEPTOR(char*, strcpy, char *to, const char *from) {  // NOLINT
#if SANITIZER_MAC
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
  if (!asan_inited) return internal_strdup(s);
  ENSURE_ASAN_INITED();
  uptr length = REAL(strlen)(s);
  if (flags()->replace_str) {
    ASAN_READ_RANGE(s, length + 1);
  }
  GET_STACK_TRACE_MALLOC;
  void *new_mem = asan_malloc(length + 1, &stack);
  REAL(memcpy)(new_mem, s, length + 1);
  return reinterpret_cast<char*>(new_mem);
}
#endif

INTERCEPTOR(uptr, strlen, const char *s) {
  if (!asan_inited) return internal_strlen(s);
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

INTERCEPTOR(uptr, wcslen, const wchar_t *s) {
  uptr length = REAL(wcslen)(s);
  if (!asan_init_is_running) {
    ENSURE_ASAN_INITED();
    ASAN_READ_RANGE(s, (length + 1) * sizeof(wchar_t));
  }
  return length;
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
  CHECK(endptr);
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
#if SANITIZER_MAC
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
#if SANITIZER_MAC
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

static void AtCxaAtexit(void *unused) {
  (void)unused;
  StopInitOrderChecking();
}

#if ASAN_INTERCEPT___CXA_ATEXIT
INTERCEPTOR(int, __cxa_atexit, void (*func)(void *), void *arg,
            void *dso_handle) {
#if SANITIZER_MAC
  if (!asan_inited) return REAL(__cxa_atexit)(func, arg, dso_handle);
#endif
  ENSURE_ASAN_INITED();
  int res = REAL(__cxa_atexit)(func, arg, dso_handle);
  REAL(__cxa_atexit)(AtCxaAtexit, 0, 0);
  return res;
}
#endif  // ASAN_INTERCEPT___CXA_ATEXIT

#if SANITIZER_WINDOWS
INTERCEPTOR_WINAPI(DWORD, CreateThread,
                   void* security, uptr stack_size,
                   DWORD (__stdcall *start_routine)(void*), void* arg,
                   DWORD thr_flags, void* tid) {
  // Strict init-order checking in thread-hostile.
  if (flags()->strict_init_order)
    StopInitOrderChecking();
  GET_STACK_TRACE_THREAD;
  u32 current_tid = GetCurrentTidOrInvalid();
  AsanThread *t = AsanThread::Create(start_routine, arg);
  CreateThreadContextArgs args = { t, &stack };
  bool detached = false;  // FIXME: how can we determine it on Windows?
  asanThreadRegistry().CreateThread(*(uptr*)t, detached, current_tid, &args);
  return REAL(CreateThread)(security, stack_size,
                            asan_thread_start, t, thr_flags, tid);
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
  SANITIZER_COMMON_INTERCEPTORS_INIT;

  // Intercept mem* functions.
  ASAN_INTERCEPT_FUNC(memcmp);
  ASAN_INTERCEPT_FUNC(memmove);
  ASAN_INTERCEPT_FUNC(memset);
  if (PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE) {
    ASAN_INTERCEPT_FUNC(memcpy);
  }

  // Intercept str* functions.
  ASAN_INTERCEPT_FUNC(strcat);  // NOLINT
  ASAN_INTERCEPT_FUNC(strchr);
  ASAN_INTERCEPT_FUNC(strcpy);  // NOLINT
  ASAN_INTERCEPT_FUNC(strlen);
  ASAN_INTERCEPT_FUNC(wcslen);
  ASAN_INTERCEPT_FUNC(strncat);
  ASAN_INTERCEPT_FUNC(strncpy);
#if ASAN_INTERCEPT_STRDUP
  ASAN_INTERCEPT_FUNC(strdup);
#endif
#if ASAN_INTERCEPT_STRNLEN
  ASAN_INTERCEPT_FUNC(strnlen);
#endif
#if ASAN_INTERCEPT_INDEX && ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX
  ASAN_INTERCEPT_FUNC(index);
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
#if ASAN_INTERCEPT_SWAPCONTEXT
  ASAN_INTERCEPT_FUNC(swapcontext);
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

  // Intercept atexit function.
#if ASAN_INTERCEPT___CXA_ATEXIT
  ASAN_INTERCEPT_FUNC(__cxa_atexit);
#endif

  // Some Windows-specific interceptors.
#if SANITIZER_WINDOWS
  InitializeWindowsInterceptors();
#endif

  VReport(1, "AddressSanitizer: libc interceptors initialized\n");
}

}  // namespace __asan
