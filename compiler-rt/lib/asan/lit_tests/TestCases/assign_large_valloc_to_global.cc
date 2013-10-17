// Make sure we don't report a leak nor hang.
// RUN: %clangxx_asan -O3 %s -o %t && %t
#if defined(__APPLE__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif
int *p = (int*)valloc(1 << 20);
int main() { }
