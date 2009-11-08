// RUN: clang-cc -verify -fsyntax-only %s
// RUN: clang-cc -emit-llvm -o %t %s

#include <stdlib.h>

int no_vars __attribute((malloc)); // expected-warning {{functions returning a pointer type}}

void  returns_void  (void) __attribute((malloc)); // expected-warning {{functions returning a pointer type}}
int   returns_int   (void) __attribute((malloc)); // expected-warning {{functions returning a pointer type}}
int * returns_intptr(void) __attribute((malloc)); // no-warning
typedef int * iptr;
iptr  returns_iptr  (void) __attribute((malloc)); // no-warning

__attribute((malloc)) void *(*f)(); //  expected-warning{{'malloc' attribute only applies to functions returning a pointer type}}
__attribute((malloc)) int (*g)(); // expected-warning{{'malloc' attribute only applies to functions returning a pointer type}}

__attribute((malloc))
void * xalloc(unsigned n) { return malloc(n); } // no-warning
// RUN: grep 'define noalias .* @xalloc(' %t

#define malloc_like __attribute((__malloc__))
void * xalloc2(unsigned) malloc_like;
void * xalloc2(unsigned n) { return malloc(n); }
// RUN: grep 'define noalias .* @xalloc2(' %t

