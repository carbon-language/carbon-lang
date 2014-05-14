// Make sure we don't report a leak nor hang.
// RUN: %clangxx_asan -O3 %s -o %t && %run %t
#include <stdlib.h>
#ifndef __APPLE__
# include <malloc.h>
#endif  // __APPLE__
int *p = (int*)valloc(1 << 20);
int main() { }
