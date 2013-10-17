// Make sure we don't report a leak nor hang.
// RUN: %clangxx_asan -O3 %s -o %t && %t
#include <malloc.h>
int *p = (int*)valloc(1 << 20);
int main() { }
