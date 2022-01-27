// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

// FIXME: Doesn't work with DLLs
// XFAIL: win32-dynamic-asan

#include <stdio.h>
#include <stdlib.h>

extern "C"
void __asan_on_error() {
  fprintf(stderr, "__asan_on_error called\n");
  fflush(stderr);
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: __asan_on_error called
}
