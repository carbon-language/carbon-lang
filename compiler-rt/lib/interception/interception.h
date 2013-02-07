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

#include "sanitizer_common/sanitizer_internal_defs.h"

// These typedefs should be used only in the interceptor definitions to replace
// the standard system types (e.g. SSIZE_T instead of ssize_t)
typedef __sanitizer::uptr SIZE_T;
typedef __sanitizer::sptr SSIZE_T;
typedef __sanitizer::sptr PTRDIFF_T;
typedef __sanitizer::s64  INTMAX_T;
// WARNING: OFF_T may be different from OS type off_t, depending on the value of
// _FILE_OFFSET_BITS. This definition of OFF_T matches the ABI of system calls
// like pread and mmap, as opposed to pread64 and mmap64.
// Mac is special.
#ifdef __APPLE__
typedef __sanitizer::u64 OFF_T;
#else
typedef __sanitizer::uptr OFF_T;
#endif
typedef __sanitizer::u64  OFF64_T;

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
//      3a) add DECLARE_REAL(int, foo, const char*, double) to a
//          header file.
// However, if the call "INTERCEPT_FUNCTION(foo)" and definition for
// INTERCEPTOR(..., foo, ...) are in different files, you'll instead need to:
//      3b) add DECLARE_REAL_AND_INTERCEPTOR(int, foo, const char*, double)
//          to a header file.

// Notes: 1. Things may not work properly if macro INTERCEPT(...) {...} or
//           DECLARE_REAL(...) are located inside namespaces.
//        2. On Mac you can also use: "OVERRIDE_FUNCTION(foo, zoo);" to
//           effectively redirect calls from "foo" to "zoo". In this case
//           you aren't required to implement
//           INTERCEPTOR(int, foo, const char *bar, double baz) {...}
//           but instead you'll have to add
//           DEFINE_REAL(int, foo, const char *bar, double baz) in your
//           source file (to define a pointer to overriden function).

// How it works:
// To replace system functions on Linux we just need to declare functions
// with same names in our library and then obtain the real function pointers
// using dlsym().
// There is one complication. A user may also intercept some of the functions
// we intercept. To resolve this we declare our interceptors with __interceptor_
// prefix, and then make actual interceptors weak aliases to __interceptor_
// functions.
// This is not so on Mac OS, where the two-level namespace makes
// our replacement functions invisible to other libraries. This may be overcomed
// using the DYLD_FORCE_FLAT_NAMESPACE, but some errors loading the shared
// libraries in Chromium were noticed when doing so.
// Instead we create a dylib containing a __DATA,__interpose section that
// associates library functions with their wrappers. When this dylib is
// preloaded before an executable using DYLD_INSERT_LIBRARIES, it routes all
// the calls to interposed functions done through stubs to the wrapper
// functions.

#if defined(__APPLE__)
# define WRAP(x) wrap_##x
# define WRAPPER_NAME(x) "wrap_"#x
# define INTERCEPTOR_ATTRIBUTE
# define DECLARE_WRAPPER(ret_type, func, ...)
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
# define DECLARE_WRAPPER(ret_type, func, ...)
#else
# define WRAP(x) __interceptor_ ## x
# define WRAPPER_NAME(x) "__interceptor_" #x
# define INTERCEPTOR_ATTRIBUTE __attribute__((visibility("default")))
# define DECLARE_WRAPPER(ret_type, func, ...) \
    extern "C" ret_type func(__VA_ARGS__) \
    __attribute__((weak, alias("__interceptor_" #func), visibility("default")));
#endif

#if !defined(__APPLE__)
# define PTR_TO_REAL(x) real_##x
# define REAL(x) __interception::PTR_TO_REAL(x)
# define FUNC_TYPE(x) x##_f

# define DECLARE_REAL(ret_type, func, ...) \
    typedef ret_type (*FUNC_TYPE(func))(__VA_ARGS__); \
    namespace __interception { \
      extern FUNC_TYPE(func) PTR_TO_REAL(func); \
    }
#else  // __APPLE__
# define REAL(x) x
# define DECLARE_REAL(ret_type, func, ...) \
    extern "C" ret_type func(__VA_ARGS__);
#endif  // __APPLE__

#define DECLARE_REAL_AND_INTERCEPTOR(ret_type, func, ...) \
  DECLARE_REAL(ret_type, func, __VA_ARGS__) \
  extern "C" ret_type WRAP(func)(__VA_ARGS__);

// Generally, you don't need to use DEFINE_REAL by itself, as INTERCEPTOR
// macros does its job. In exceptional cases you may need to call REAL(foo)
// without defining INTERCEPTOR(..., foo, ...). For example, if you override
// foo with an interceptor for other function.
#if !defined(__APPLE__)
# define DEFINE_REAL(ret_type, func, ...) \
    typedef ret_type (*FUNC_TYPE(func))(__VA_ARGS__); \
    namespace __interception { \
      FUNC_TYPE(func) PTR_TO_REAL(func); \
    }
#else
# define DEFINE_REAL(ret_type, func, ...)
#endif

#define INTERCEPTOR(ret_type, func, ...) \
  DEFINE_REAL(ret_type, func, __VA_ARGS__) \
  DECLARE_WRAPPER(ret_type, func, __VA_ARGS__) \
  extern "C" \
  INTERCEPTOR_ATTRIBUTE \
  ret_type WRAP(func)(__VA_ARGS__)

#if defined(_WIN32)
# define INTERCEPTOR_WINAPI(ret_type, func, ...) \
    typedef ret_type (__stdcall *FUNC_TYPE(func))(__VA_ARGS__); \
    namespace __interception { \
      FUNC_TYPE(func) PTR_TO_REAL(func); \
    } \
    DECLARE_WRAPPER(ret_type, func, __VA_ARGS__) \
    extern "C" \
    INTERCEPTOR_ATTRIBUTE \
    ret_type __stdcall WRAP(func)(__VA_ARGS__)
#endif

// ISO C++ forbids casting between pointer-to-function and pointer-to-object,
// so we use casting via an integral type __interception::uptr,
// assuming that system is POSIX-compliant. Using other hacks seem
// challenging, as we don't even pass function type to
// INTERCEPT_FUNCTION macro, only its name.
namespace __interception {
#if defined(_WIN64)
typedef unsigned long long uptr;  // NOLINT
#else
typedef unsigned long uptr;  // NOLINT
#endif  // _WIN64
}  // namespace __interception

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
