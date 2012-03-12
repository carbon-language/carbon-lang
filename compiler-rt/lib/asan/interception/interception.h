//===-- interception.h ------------------------------------------*- C++ -*-===//
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
// Machinery for providing replacements/wrappers for system functions.
//===----------------------------------------------------------------------===//

#ifndef INTERCEPTION_H
#define INTERCEPTION_H

#if !defined(__linux__) && !defined(__APPLE__) && !defined(_WIN32)
# error "Interception doesn't work on this operating system."
#endif

// How to use this library:
//      1) Include this header to define your own interceptors
//         (see details below).
//      2) Build all *.cc files and link against them.
// On Mac you will also need to:
//      3) Provide your own implementation for the following functions:
//           mach_error_t __interception::allocate_island(void **ptr,
//                                                      size_t size,
//                                                      void *hint);
//           mach_error_t __interception::deallocate_island(void *ptr);
//         See "interception_mac.h" for more details.

// How to add an interceptor:
// Suppose you need to wrap/replace system function (generally, from libc):
//      int foo(const char *bar, double baz);
// You'll need to:
//      1) define INTERCEPTOR(int, foo, const char *bar, double baz) { ... } in
//         your source file.
//      2) Call "INTERCEPT_FUNCTION(foo)" prior to the first call of "foo".
//         INTERCEPT_FUNCTION(foo) evaluates to "true" iff the function was
//         intercepted successfully.
// You can access original function by calling REAL(foo)(bar, baz).
// By default, REAL(foo) will be visible only inside your interceptor, and if
// you want to use it in other parts of RTL, you'll need to:
//      3a) add DECLARE_REAL(int, foo, const char*, double); to a
//          header file.
// However, if the call "INTERCEPT_FUNCTION(foo)" and definition for
// INTERCEPTOR(..., foo, ...) are in different files, you'll instead need to:
//      3b) add DECLARE_REAL_AND_INTERCEPTOR(int, foo, const char*, double);
//          to a header file.

// Notes: 1. Things may not work properly if macro INTERCEPT(...) {...} or
//           DECLARE_REAL(...); are located inside namespaces.
//        2. On Mac you can also use: "OVERRIDE_FUNCTION(foo, zoo);" to
//           effectively redirect calls from "foo" to "zoo". In this case
//           you aren't required to implement
//           INTERCEPTOR(int, foo, const char *bar, double baz);
//           but instead you'll have to add
//           DEFINE_REAL(int, foo, const char *bar, double baz); in your
//           source file (to define a pointer to overriden function).

// How it works:
// To replace weak system functions on Linux we just need to declare functions
// with same names in our library and then obtain the real function pointers
// using dlsym(). This is not so on Mac OS, where the two-level namespace makes
// our replacement functions invisible to other libraries. This may be overcomed
// using the DYLD_FORCE_FLAT_NAMESPACE, but some errors loading the shared
// libraries in Chromium were noticed when doing so.
// Instead we use mach_override, a handy framework for patching functions at
// runtime. To avoid possible name clashes, our replacement functions have
// the "wrap_" prefix on Mac.

#if defined(__APPLE__)
# define WRAP(x) wrap_##x
# define WRAPPER_NAME(x) "wrap_"#x
# define INTERCEPTOR_ATTRIBUTE
#elif defined(_WIN32)
# if defined(_DLL)  // DLL CRT
#  define WRAP(x) x
#  define WRAPPER_NAME(x) #x
#  define INTERCEPTOR_ATTRIBUTE
# else  // Static CRT
#  define WRAP(x) wrap_##x
#  define WRAPPER_NAME(x) "wrap_"#x
#  define INTERCEPTOR_ATTRIBUTE
# endif
#else
# define WRAP(x) x
# define WRAPPER_NAME(x) #x
# define INTERCEPTOR_ATTRIBUTE __attribute__((visibility("default")))
#endif

#define PTR_TO_REAL(x) real_##x
#define REAL(x) __interception::PTR_TO_REAL(x)
#define FUNC_TYPE(x) x##_f

#define DECLARE_REAL(ret_type, func, ...); \
  typedef ret_type (*FUNC_TYPE(func))(__VA_ARGS__); \
  namespace __interception { \
    extern FUNC_TYPE(func) PTR_TO_REAL(func); \
  }

#define DECLARE_REAL_AND_INTERCEPTOR(ret_type, func, ...); \
  DECLARE_REAL(ret_type, func, ##__VA_ARGS__); \
  extern "C" ret_type WRAP(func)(__VA_ARGS__);

// FIXME(timurrrr): We might need to add DECLARE_REAL_EX etc to support
// different calling conventions later.

#define DEFINE_REAL_EX(ret_type, convention, func, ...); \
  typedef ret_type (convention *FUNC_TYPE(func))(__VA_ARGS__); \
  namespace __interception { \
    FUNC_TYPE(func) PTR_TO_REAL(func); \
  }

// Generally, you don't need to use DEFINE_REAL by itself, as INTERCEPTOR
// macros does its job. In exceptional cases you may need to call REAL(foo)
// without defining INTERCEPTOR(..., foo, ...). For example, if you override
// foo with an interceptor for other function.
#define DEFAULT_CONVENTION

#define DEFINE_REAL(ret_type, func, ...); \
  DEFINE_REAL_EX(ret_type, DEFAULT_CONVENTION, func, __VA_ARGS__);

#define INTERCEPTOR_EX(ret_type, convention, func, ...) \
  DEFINE_REAL_EX(ret_type, convention, func, __VA_ARGS__); \
  extern "C" \
  INTERCEPTOR_ATTRIBUTE \
  ret_type convention WRAP(func)(__VA_ARGS__)

#define INTERCEPTOR(ret_type, func, ...) \
  INTERCEPTOR_EX(ret_type, DEFAULT_CONVENTION, func, __VA_ARGS__)

#if defined(_WIN32)
# define INTERCEPTOR_WINAPI(ret_type, func, ...) \
  INTERCEPTOR_EX(ret_type, __stdcall, func, __VA_ARGS__)
#endif

#define INCLUDED_FROM_INTERCEPTION_LIB

#if defined(__linux__)
# include "interception_linux.h"
# define INTERCEPT_FUNCTION(func) INTERCEPT_FUNCTION_LINUX(func)
#elif defined(__APPLE__)
# include "interception_mac.h"
# define OVERRIDE_FUNCTION(old_func, new_func) \
    OVERRIDE_FUNCTION_MAC(old_func, new_func)
# define INTERCEPT_FUNCTION(func) INTERCEPT_FUNCTION_MAC(func)
#else  // defined(_WIN32)
# include "interception_win.h"
# define INTERCEPT_FUNCTION(func) INTERCEPT_FUNCTION_WIN(func)
#endif

#undef INCLUDED_FROM_INTERCEPTION_LIB

#endif  // INTERCEPTION_H
