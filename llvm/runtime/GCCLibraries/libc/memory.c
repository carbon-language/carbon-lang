//===-- memory.c - String functions for the LLVM libc Library ----*- C -*-===//
// 
// A lot of this code is ripped gratuitously from glibc and libiberty.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

void *malloc(size_t) __attribute__((weak));
void free(void *) __attribute__((weak));
void *memset(void *, int, size_t) __attribute__((weak));
void *calloc(size_t nelem, size_t elsize) __attribute__((weak));

void *calloc(size_t nelem, size_t elsize) {
  void *Result = malloc(nelem*elsize);
  return memset(Result, 0, nelem*elsize);
}
