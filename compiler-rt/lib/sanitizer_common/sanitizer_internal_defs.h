//===-- sanitizer_internal_defs.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
// It contains macro used in run-time libraries code.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_DEFS_H
#define SANITIZER_DEFS_H

#include "sanitizer_interface_defs.h"
using namespace __sanitizer;  // NOLINT
// ----------- ATTENTION -------------
// This header should NOT include any other headers to avoid portability issues.

// Common defs.
#define INLINE static inline
#define INTERFACE_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
#define WEAK SANITIZER_WEAK_ATTRIBUTE

// Platform-specific defs.
#if defined(_WIN32)
typedef unsigned long    DWORD;  // NOLINT
# define ALWAYS_INLINE __declspec(forceinline)
// FIXME(timurrrr): do we need this on Windows?
# define ALIAS(x)
# define ALIGNED(x) __declspec(align(x))
# define FORMAT(f, a)
# define NOINLINE __declspec(noinline)
# define NORETURN __declspec(noreturn)
# define THREADLOCAL   __declspec(thread)
# define NOTHROW
#else  // _WIN32
# define ALWAYS_INLINE __attribute__((always_inline))
# define ALIAS(x) __attribute__((alias(x)))
# define ALIGNED(x) __attribute__((aligned(x)))
# define FORMAT(f, a)  __attribute__((format(printf, f, a)))
# define NOINLINE __attribute__((noinline))
# define NORETURN  __attribute__((noreturn))
# define THREADLOCAL   __thread
# ifdef __cplusplus
#   define NOTHROW throw()
# else
#   define NOTHROW __attribute__((__nothrow__))
#endif
#endif  // _WIN32

// We have no equivalent of these on Windows.
#ifndef _WIN32
# define LIKELY(x)     __builtin_expect(!!(x), 1)
# define UNLIKELY(x)   __builtin_expect(!!(x), 0)
# define UNUSED __attribute__((unused))
# define USED __attribute__((used))
#endif

#if defined(_WIN32)
typedef DWORD thread_return_t;
# define THREAD_CALLING_CONV __stdcall
#else  // _WIN32
typedef void* thread_return_t;
# define THREAD_CALLING_CONV
#endif  // _WIN32
typedef thread_return_t (THREAD_CALLING_CONV *thread_callback_t)(void* arg);

// If __WORDSIZE was undefined by the platform, define it in terms of the
// compiler built-ins __LP64__ and _WIN64.
#ifndef __WORDSIZE
# if __LP64__ || defined(_WIN64)
#  define __WORDSIZE 64
# else
#  define __WORDSIZE 32
#  endif
#endif  // __WORDSIZE

// NOTE: Functions below must be defined in each run-time.
namespace __sanitizer {
void NORETURN Die();
void NORETURN CheckFailed(const char *file, int line, const char *cond,
                          u64 v1, u64 v2);
}  // namespace __sanitizer

// Check macro
#define RAW_CHECK_MSG(expr, msg) do { \
  if (!(expr)) { \
    RawWrite(msg); \
    Die(); \
  } \
} while (0)

#define RAW_CHECK(expr) RAW_CHECK_MSG(expr, #expr)

#define CHECK_IMPL(c1, op, c2) \
  do { \
    __sanitizer::u64 v1 = (u64)(c1); \
    __sanitizer::u64 v2 = (u64)(c2); \
    if (!(v1 op v2)) \
      __sanitizer::CheckFailed(__FILE__, __LINE__, \
        "(" #c1 ") " #op " (" #c2 ")", v1, v2); \
  } while (false) \
/**/

#define CHECK(a)       CHECK_IMPL((a), !=, 0)
#define CHECK_EQ(a, b) CHECK_IMPL((a), ==, (b))
#define CHECK_NE(a, b) CHECK_IMPL((a), !=, (b))
#define CHECK_LT(a, b) CHECK_IMPL((a), <,  (b))
#define CHECK_LE(a, b) CHECK_IMPL((a), <=, (b))
#define CHECK_GT(a, b) CHECK_IMPL((a), >,  (b))
#define CHECK_GE(a, b) CHECK_IMPL((a), >=, (b))

#define UNIMPLEMENTED() CHECK("unimplemented" && 0)

#define COMPILER_CHECK(pred) IMPL_COMPILER_ASSERT(pred, __LINE__)

#define IMPL_PASTE(a, b) a##b
#define IMPL_COMPILER_ASSERT(pred, line) \
    typedef char IMPL_PASTE(assertion_failed_##_, line)[2*(int)(pred)-1];

// Limits for integral types. We have to redefine it in case we don't
// have stdint.h (like in Visual Studio 9).
#if __WORDSIZE == 64
# define __INT64_C(c)  c ## L
# define __UINT64_C(c) c ## UL
#else
# define __INT64_C(c)  c ## LL
# define __UINT64_C(c) c ## ULL
#endif  // __WORDSIZE == 64
#undef INT32_MIN
#define INT32_MIN              (-2147483647-1)
#undef INT32_MAX
#define INT32_MAX              (2147483647)
#undef UINT32_MAX
#define UINT32_MAX             (4294967295U)
#undef INT64_MIN
#define INT64_MIN              (-__INT64_C(9223372036854775807)-1)
#undef INT64_MAX
#define INT64_MAX              (__INT64_C(9223372036854775807))
#undef UINT64_MAX
#define UINT64_MAX             (__UINT64_C(18446744073709551615))

#endif  // SANITIZER_DEFS_H
