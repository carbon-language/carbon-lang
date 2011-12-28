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

// To replace weak system functions on Linux we just need to declare functions
// with same names in our library and then obtain the real function pointers
// using dlsym(). This is not so on Mac OS, where the two-level namespace makes
// our replacement functions invisible to other libraries. This may be overcomed
// using the DYLD_FORCE_FLAT_NAMESPACE, but some errors loading the shared
// libraries in Chromium were noticed when doing so.
// Instead we use mach_override, a handy framework for patching functions at
// runtime. To avoid possible name clashes, our replacement functions have
// the "wrap_" prefix on Mac.
//
// After interception, the calls to system functions will be substituted by
// calls to our interceptors. We store pointers to system function f()
// in __asan::real_f().
//
// TODO(glider): mach_override_ptr() tends to spend too much time
// in allocateBranchIsland(). This should be ok for real-word
// application, but slows down our tests which fork too many children.
#ifdef __APPLE__
#include "mach_override/mach_override.h"
#define WRAP(x) wrap_##x
#define WRAPPER_NAME(x) "wrap_"#x

#define OVERRIDE_FUNCTION(oldfunc, newfunc)                             \
  CHECK(0 == __asan_mach_override_ptr((void*)(oldfunc),                        \
                                      (void*)(newfunc),                        \
                                      (void**)&real_##oldfunc));               \
  CHECK(real_##oldfunc != NULL);

#define OVERRIDE_FUNCTION_IF_EXISTS(oldfunc, newfunc)                   \
  do { __asan_mach_override_ptr((void*)(oldfunc),                              \
                                (void*)(newfunc),                              \
                                (void**)&real_##oldfunc); } while (0)

#define INTERCEPT_FUNCTION(func)                                        \
  OVERRIDE_FUNCTION(func, WRAP(func))

#define INTERCEPT_FUNCTION_IF_EXISTS(func)                              \
  OVERRIDE_FUNCTION_IF_EXISTS(func, WRAP(func))

#else  // __linux__
#define WRAP(x) x
#define WRAPPER_NAME(x) #x

#define INTERCEPT_FUNCTION(func)                                        \
  CHECK((real_##func = (func##_f)dlsym(RTLD_NEXT, #func)));

#define INTERCEPT_FUNCTION_IF_EXISTS(func)                              \
  do { real_##func = (func##_f)dlsym(RTLD_NEXT, #func); } while (0)
#endif

#ifdef __APPLE__
int WRAP(memcmp)(const void *a1, const void *a2, size_t size);
void *WRAP(memcpy)(void *to, const void *from, size_t size);
void *WRAP(memmove)(void *to, const void *from, size_t size);
void *WRAP(memset)(void *block, int c, size_t size);
int WRAP(strcasecmp)(const char *s1, const char *s2);
char *WRAP(strcat)(char *to, const char *from);  // NOLINT
char *WRAP(strchr)(const char *string, int c);
int WRAP(strcmp)(const char *s1, const char *s2);
char *WRAP(strcpy)(char *to, const char *from);  // NOLINT
char *WRAP(strdup)(const char *s);
size_t WRAP(strlen)(const char *s);
int WRAP(strncasecmp)(const char *s1, const char *s2, size_t n);
int WRAP(strncmp)(const char *s1, const char *s2, size_t size);
char *WRAP(strncpy)(char *to, const char *from, size_t size);
#endif

namespace __asan {

typedef void* (*index_f)(const char *string, int c);
typedef int (*memcmp_f)(const void *a1, const void *a2, size_t size);
typedef void* (*memcpy_f)(void *to, const void *from, size_t size);
typedef void* (*memmove_f)(void *to, const void *from, size_t size);
typedef void* (*memset_f)(void *block, int c, size_t size);
typedef int (*strcasecmp_f)(const char *s1, const char *s2);
typedef char* (*strcat_f)(char *to, const char *from);
typedef char* (*strchr_f)(const char *str, int c);
typedef int (*strcmp_f)(const char *s1, const char *s2);
typedef char* (*strcpy_f)(char *to, const char *from);
typedef char* (*strdup_f)(const char *s);
typedef size_t (*strlen_f)(const char *s);
typedef int (*strncasecmp_f)(const char *s1, const char *s2, size_t n);
typedef int (*strncmp_f)(const char *s1, const char *s2, size_t size);
typedef char* (*strncpy_f)(char *to, const char *from, size_t size);
typedef size_t (*strnlen_f)(const char *s, size_t maxlen);

// __asan::real_X() holds pointer to library implementation of X().
extern index_f          real_index;
extern memcmp_f         real_memcmp;
extern memcpy_f         real_memcpy;
extern memmove_f        real_memmove;
extern memset_f         real_memset;
extern strcasecmp_f     real_strcasecmp;
extern strcat_f         real_strcat;
extern strchr_f         real_strchr;
extern strcmp_f         real_strcmp;
extern strcpy_f         real_strcpy;
extern strdup_f         real_strdup;
extern strlen_f         real_strlen;
extern strncasecmp_f    real_strncasecmp;
extern strncmp_f        real_strncmp;
extern strncpy_f        real_strncpy;
extern strnlen_f        real_strnlen;

// __asan::internal_X() is the implementation of X() for use in RTL.
size_t internal_strlen(const char *s);
size_t internal_strnlen(const char *s, size_t maxlen);

// Initializes pointers to str*/mem* functions.
void InitializeAsanInterceptors();

}  // namespace __asan

#endif  // ASAN_INTERCEPTORS_H
