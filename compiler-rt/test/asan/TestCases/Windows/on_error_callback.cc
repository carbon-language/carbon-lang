// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

// FIXME: merge this with the common on_error_callback test when we can run
// common tests on Windows.

#include <stdio.h>
#include <stdlib.h>

extern "C"
void __asan_on_error() {
  fprintf(stderr, "__asan_on_error called");
  fflush(0);
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: __asan_on_error called
}
