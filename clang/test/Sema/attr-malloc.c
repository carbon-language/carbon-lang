// RUN: clang-cc -verify -fsyntax-only %s &&
// RUN: clang-cc -emit-llvm -o %t %s &&

#include <stdlib.h>

int no_vars __attribute((malloc)); // expected-warning {{only applies to function types}}

__attribute((malloc))
void * xalloc(unsigned n) { return malloc(n); }
// RUN: grep 'define noalias .* @xalloc(' %t &&

#define __malloc_like __attribute((__malloc__))
void * xalloc2(unsigned) __malloc_like;
void * xalloc2(unsigned n) { return malloc(n); }
// RUN: grep 'define noalias .* @xalloc2(' %t

