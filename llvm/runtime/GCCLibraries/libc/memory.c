//===-- memory.c - String functions for the LLVM libc Library ----*- C -*-===//
// 
// A lot of this code is ripped gratuitously from glibc and libiberty.
//
//===---------------------------------------------------------------------===//

#include <stdlib.h>

// If we're not being compiled with GCC, turn off attributes. Question is how
// to handle overriding of memory allocation functions in that case.
#ifndef __GNUC__
#define __attribute__(X)
#endif
    
// For now, turn off the weak linkage attribute on Mac OS X.
#if defined(__GNUC__) && defined(__APPLE_CC__)
#define __ATTRIBUTE_WEAK__
#elif defined(__GNUC__)
#define __ATTRIBUTE_WEAK__ __attribute__((weak))
#else
#define __ATTRIBUTE_WEAK__
#endif

void *malloc(size_t) __ATTRIBUTE_WEAK__;
void free(void *) __ATTRIBUTE_WEAK__;
void *memset(void *, int, size_t) __ATTRIBUTE_WEAK__;
void *calloc(size_t nelem, size_t elsize) __ATTRIBUTE_WEAK__;

void *calloc(size_t nelem, size_t elsize) {
  void *Result = malloc(nelem*elsize);
  return memset(Result, 0, nelem*elsize);
}
