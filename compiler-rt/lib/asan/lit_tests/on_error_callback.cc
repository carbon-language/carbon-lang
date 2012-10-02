// RUN: %clangxx_asan -O2 %s -o %t && %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

extern "C"
void __asan_on_error() {
  fprintf(stderr, "__asan_on_error called");
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: __asan_on_error called
}
