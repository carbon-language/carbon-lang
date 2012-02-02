//===-- asan_interceptors.h -------------------------------------*- C++ -*-===//
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
// ASan-private header for asan_interceptors.cc
//===----------------------------------------------------------------------===//
#ifndef ASAN_INTERCEPTORS_H
#define ASAN_INTERCEPTORS_H

#include "asan_internal.h"

// Suppose you need to wrap/replace system function (generally, from libc):
//      int foo(const char *bar, double baz);
// You'll need to:
//      1) define INTERCEPT(int, foo, const char *bar, double baz) { ... }
//      2) add a line "INTERCEPT_FUNCTION(foo)" to InitializeAsanInterceptors()
// You can access original function by calling __asan::real_foo(bar, baz).
// By defualt, real_foo will be visible only inside your interceptor, and if
// you want to use it in other parts of RTL, you'll need to:
//      3a) add DECLARE_REAL(int, foo, const char*, double); to a
//          header file.
// However, if you want to implement your interceptor somewhere outside
// asan_interceptors.cc, you'll instead need to:
//      3b) add DECLARE_REAL_AND_INTERCEPTOR(int, foo, const char*, double);
//          to a header.

#if defined(__APPLE__)
# define WRAP(x) wrap_##x
# define WRAPPER_NAME(x) "wrap_"#x
# define INTERCEPTOR_ATTRIBUTE
#elif defined(_WIN32)
// TODO(timurrrr): we're likely to use something else later on Windows.
# define WRAP(x) wrap_##x
# define WRAPPER_NAME(x) #x
# define INTERCEPTOR_ATTRIBUTE
#else
# define WRAP(x) x
# define WRAPPER_NAME(x) #x
# define INTERCEPTOR_ATTRIBUTE __attribute__((visibility("default")))
#endif

#define REAL(x) real_##x
#define FUNC_TYPE(x) x##_f

#define DECLARE_REAL(ret_type, func, ...); \
  typedef ret_type (*FUNC_TYPE(func))(__VA_ARGS__); \
  namespace __asan { \
    extern FUNC_TYPE(func) REAL(func); \
  }

#define DECLARE_REAL_AND_INTERCEPTOR(ret_type, func, ...); \
    DECLARE_REAL(ret_type, func, ##__VA_ARGS__); \
    extern "C" \
    ret_type WRAP(func)(__VA_ARGS__);

// Generally, you don't need to use DEFINE_REAL by itself, as INTERCEPTOR
// macros does its job. In exceptional cases you may need to call REAL(foo)
// without defining INTERCEPTOR(..., foo, ...). For example, if you override
// foo with interceptor for other function.
#define DEFINE_REAL(ret_type, func, ...); \
  typedef ret_type (*FUNC_TYPE(func))(__VA_ARGS__); \
  namespace __asan { \
    FUNC_TYPE(func) REAL(func); \
  }

#define INTERCEPTOR(ret_type, func, ...); \
  DEFINE_REAL(ret_type, func, __VA_ARGS__); \
  extern "C" \
  INTERCEPTOR_ATTRIBUTE \
  ret_type WRAP(func)(__VA_ARGS__)

DECLARE_REAL(int, memcmp, const void *a1, const void *a2, size_t size);
DECLARE_REAL(void*, memcpy, void *to, const void *from, size_t size);
DECLARE_REAL(void*, memset, void *block, int c, size_t size);
DECLARE_REAL(char*, strchr, const char *str, int c);
DECLARE_REAL(size_t, strlen, const char *s);
DECLARE_REAL(char*, strncpy, char *to, const char *from, size_t size);
DECLARE_REAL(size_t, strnlen, const char *s, size_t maxlen);
struct sigaction;
DECLARE_REAL(int, sigaction, int signum, const struct sigaction *act,
                             struct sigaction *oldact);

namespace __asan {

// __asan::internal_X() is the implementation of X() for use in RTL.
size_t internal_strlen(const char *s);
size_t internal_strnlen(const char *s, size_t maxlen);
void* internal_memchr(const void* s, int c, size_t n);
int internal_memcmp(const void* s1, const void* s2, size_t n);
char *internal_strstr(const char *haystack, const char *needle);
char *internal_strncat(char *dst, const char *src, size_t n);
int internal_strcmp(const char *s1, const char *s2);

void InitializeAsanInterceptors();

}  // namespace __asan

#endif  // ASAN_INTERCEPTORS_H
