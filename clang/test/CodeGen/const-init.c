// RUN: clang -emit-llvm %s 2>&1 | not grep warning

#include <stdint.h>

// Brace-enclosed string array initializers
char a[] = { "asdf" };

// Double-implicit-conversions of array/functions (not legal C, but
// clang accepts it for gcc compat).
intptr_t b = a;
int c();
void *d = c;
intptr_t e = c;

int f, *g = __extension__ &f, *h = (1 != 1) ? &f : &f;
