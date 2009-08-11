// RUN: clang-cc -verify -fsyntax-only %s &&
// RUN: clang-cc -emit-llvm -o %t %s &&

#include <stdlib.h>

int no_vars __attribute((malloc)); // expected-warning {{only applies to function types}}

void  returns_void  (void) __attribute((malloc)); // expected-warning {{functions returning pointer type}}
int   returns_int   (void) __attribute((malloc)); // expected-warning {{functions returning pointer type}}
int * returns_intptr(void) __attribute((malloc));
typedef int * iptr;
iptr  returns_iptr  (void) __attribute((malloc));

__attribute((malloc))
void * xalloc(unsigned n) { return malloc(n); }
// RUN: grep 'define noalias .* @xalloc(' %t &&

#define malloc_like __attribute((__malloc__))
void * xalloc2(unsigned) malloc_like;
void * xalloc2(unsigned n) { return malloc(n); }
// RUN: grep 'define noalias .* @xalloc2(' %t

