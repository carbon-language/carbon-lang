// Make sure we don't report a leak nor hang.
// RUN: %clangxx_asan -O3 %s -o %t && %run %t
#include <stdlib.h>
#include <unistd.h>
#if !defined(__APPLE__) && !defined(__FreeBSD__)
# include <malloc.h>
#endif  // !__APPLE__ && !__FreeBSD__
int *p = (int*)valloc(1 << 20);
int main() { }
