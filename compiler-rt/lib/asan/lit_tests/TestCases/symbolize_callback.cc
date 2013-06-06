// RUN: %clangxx_asan -O2 %s -o %t && %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

extern "C"
bool __asan_symbolize(const void *pc, char *out_buffer, int out_size) {
  snprintf(out_buffer, out_size, "MySymbolizer");
  return true;
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: MySymbolizer
}
