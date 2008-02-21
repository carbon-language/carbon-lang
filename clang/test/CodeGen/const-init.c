// RUN: clang -emit-llvm %s

#include <stdint.h>

// Brace-enclosed string array initializers
char a[] = { "asdf" };

// Double-implicit-conversions of array/functions (not legal C, but
// clang accepts it for gcc compat).
intptr_t b = a;
int c();
void *d = c;
intptr_t e = c;
